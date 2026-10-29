"""Полный воспроизводимый прогон: очистка -> EDA -> walk-forward -> метрики -> устойчивость -> графики.

Запуск из корня репозитория:  python src/run_all.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import data as data_mod
from config import (DATA_RAW, DATA_CLEAN, RESULTS, FIGURES, HORIZONS, TEST_SIZE,
                    ROLLING_WINDOW, INSTRUMENT, TIMEZONE, H_MAX)
from evaluation import walk_forward, metric_table, metrics, dm_vs_baseline, test_origins
from models import all_models, Drift, ARIMA, ETS, LightGBM, GRU

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"Naive": "#444444", "Drift": "#999999", "ARIMA": "#1f77b4", "ETS": "#17becf",
          "LightGBM": "#2ca02c", "GRU": "#d62728"}


def log(*a):
    print(*a, flush=True)


def eda(df, origins):
    r = np.log(df["close"]).diff().dropna()
    adf_p = adfuller(np.log(df["close"]))[1]
    adf_r = adfuller(r)[1]
    lb = acorr_ljungbox(r, lags=[5, 10])
    st = dict(
        instrument=INSTRUMENT, timezone=TIMEZONE, n_obs=len(df),
        period=[str(df.date.min().date()), str(df.date.max().date())],
        test_period=[str(df.date.iloc[origins[0]].date()), str(df.date.iloc[origins[-1] + H_MAX].date())],
        close_min=float(df.close.min()), close_max=float(df.close.max()),
        close_first=float(df.close.iloc[0]), close_last=float(df.close.iloc[-1]),
        mean_daily_logret_pct=float(100 * r.mean()), daily_vol_pct=float(100 * r.std()),
        annual_vol_pct=float(100 * r.std() * np.sqrt(252)),
        skew=float(r.skew()), excess_kurtosis=float(r.kurt()),
        share_up_days_pct=float(100 * (r > 0).mean()),
        adf_p_logprice=float(adf_p), adf_p_returns=float(adf_r),
        ljungbox_p_lag5=float(lb["lb_pvalue"].iloc[0]), ljungbox_p_lag10=float(lb["lb_pvalue"].iloc[1]),
        acf_returns_lag1=float(r.autocorr(1)),
    )
    (RESULTS / "eda_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(df.date, df.close, color="black", lw=1)
    ax[0].axvspan(df.date.iloc[origins[0]], df.date.iloc[-1], color="orange", alpha=0.15, label="тест")
    ax[0].set_ylabel("Close, USD"); ax[0].legend(); ax[0].set_title(f"{INSTRUMENT}, дневные цены закрытия")
    ax[1].bar(df.date.iloc[1:], 100 * r.values, color="steelblue", width=1)
    ax[1].set_ylabel("Лог-доходность, %")
    fig.tight_layout(); fig.savefig(FIGURES / "01_series.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].hist(100 * r, bins=40, color="steelblue", density=True)
    ax[0].set_title("Распределение доходностей, %")
    plot_acf(r, lags=20, ax=ax[1], title="ACF доходностей")
    plot_pacf(r, lags=20, ax=ax[2], title="PACF доходностей", method="ywm")
    fig.tight_layout(); fig.savefig(FIGURES / "02_returns_acf.png"); plt.close(fig)
    return st


def plot_forecasts(pred, df, origins):
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(11, 4 * len(HORIZONS)))
    for ax, h in zip(axes, HORIZONS):
        g = pred[pred.h == h]
        fact = df.iloc[origins[0] - 20:origins[-1] + H_MAX + 1]
        ax.plot(fact.date, fact.close, color="black", lw=2, label="факт")
        for m, gm in g.groupby("model", sort=False):
            ax.plot(gm.target, gm.y_pred, lw=1.1, color=COLORS.get(m), label=m,
                    ls="--" if m in ("Naive", "Drift") else "-")
        ax.set_title(f"Прогнозы на тесте, горизонт h={h}"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "03_forecasts.png"); plt.close(fig)


def plot_metrics(mt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["MAE", "sMAPE", "DirAcc"]):
        piv = mt.pivot(index="model", columns="h", values=col).reindex(mt.model.unique())
        piv.plot.bar(ax=ax, rot=30, color=["#4c72b0", "#dd8452"])
        ax.set_title(col + (" (%)" if col != "MAE" else " (USD)"))
        if col == "DirAcc":
            ax.axhline(50, color="k", lw=0.8, ls=":")
        ax.legend(title="h")
    fig.tight_layout(); fig.savefig(FIGURES / "04_metrics.png"); plt.close(fig)


def plot_cum_error(pred):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(12, 3.8))
    for ax, h in zip(axes, HORIZONS):
        g = pred[pred.h == h].copy()
        g["ae"] = (g.y_pred - g.y_true).abs()
        base = g[g.model == "Naive"].set_index("origin")["ae"]
        for m, gm in g.groupby("model", sort=False):
            if m == "Naive":
                continue
            d = (gm.set_index("origin")["ae"] - base).cumsum()
            ax.plot(d.index, d.values, color=COLORS.get(m), label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"Накопленная разница |ошибок| с Naive, h={h}\n(ниже нуля -- лучше Naive)")
        ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "05_cum_error_vs_naive.png"); plt.close(fig)


def plot_seeds(mem_mt, ens_mt):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, h in enumerate(HORIZONS):
        v = mem_mt[mem_mt.h == h]["MAE"]
        ax.scatter([i] * len(v), v, color="#d62728", alpha=0.6, label="отдельные seed" if i == 0 else None)
        e = ens_mt[(ens_mt.model == "GRU") & (ens_mt.h == h)]["MAE"].iloc[0]
        nv = ens_mt[(ens_mt.model == "Naive") & (ens_mt.h == h)]["MAE"].iloc[0]
        ax.scatter(i, e, marker="*", s=200, color="k", label="ансамбль" if i == 0 else None)
        ax.scatter(i, nv, marker="_", s=600, color="#444", label="Naive" if i == 0 else None)
    ax.set_xticks(range(len(HORIZONS)), [f"h={h}" for h in HORIZONS]); ax.set_ylabel("MAE, USD")
    ax.set_title("GRU: разброс по seed"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "06_gru_seeds.png"); plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    if any(DATA_RAW.glob("*.csv")):
        data_mod.main()
    df = data_mod.load_clean()
    origins = test_origins(len(df))
    log(f"Наблюдений: {len(df)}, точек прогноза в тесте: {len(origins)}, "
        f"начальная обучающая выборка: {origins[0] + 1}")
    st = eda(df, origins)

    # --- основной эксперимент (расширяющееся окно)
    preds, timings, members, info = [], [], [], {}
    for model, k in all_models():
        log(f"walk-forward: {model.name} (переобучение каждые {k} точек)")
        p, tm, mem = walk_forward(model, df, origins, refit_every=k)
        preds.append(p); timings.append(tm)
        if len(mem):
            members.append(mem)
        if isinstance(model, ARIMA):
            info["ARIMA"] = dict(order=list(model.order), trend=model.trend)
            pd.DataFrame(model.grid_).sort_values("aic").to_csv(RESULTS / "tuning_arima.csv", index=False)
        if isinstance(model, LightGBM):
            info["LightGBM"] = model.params
            model.tuning_.sort_values("val_mae_logret").to_csv(RESULTS / "tuning_lightgbm.csv", index=False)
        if isinstance(model, GRU):
            info["GRU"] = dict(lookback=model.L, hidden=model.hidden, seeds=model.seeds)
            model.tuning_.sort_values("val_mae_logprice").to_csv(RESULTS / "tuning_gru.csv", index=False)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(RESULTS / "predictions.csv", index=False, float_format="%.4f")
    (RESULTS / "chosen_hyperparams.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    mt = metric_table(pred)
    naive_mae = mt[mt.model == "Naive"].set_index("h")["MAE"]
    mt["MAE_vs_Naive"] = mt.apply(lambda r: r.MAE / naive_mae[r.h], axis=1)
    mt.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.4f")
    log(mt.to_string())

    up = pred[pred.model == "Naive"].groupby("h").apply(
        lambda g: 100 * np.mean(g.y_true > g.y_last), include_groups=False)
    up.rename("share_up_pct").to_csv(RESULTS / "direction_reference.csv")

    tm = pd.DataFrame(timings)
    tm["total_s"] = tm.fit_s + tm.predict_s
    tm.to_csv(RESULTS / "timing.csv", index=False, float_format="%.3f")

    dm = dm_vs_baseline(pred)
    dm.to_csv(RESULTS / "dm_tests.csv", index=False, float_format="%.4f")

    # --- устойчивость 1: подпериоды теста
    pred["block"] = pred.groupby(["model", "h"]).cumcount() // (TEST_SIZE // 3) + 1
    sub = metric_table(pred, by=("model", "h", "block"))
    # округление: совпадающие модели (ARIMA(0,1,0) и Naive) получают одинаковый ранг
    sub["rank_MAE"] = sub.groupby(["h", "block"])["MAE"].transform(
        lambda s: s.round(6).rank(method="min"))
    sub.to_csv(RESULTS / "robust_subperiods.csv", index=False, float_format="%.4f")
    pred.drop(columns="block", inplace=True)

    # --- устойчивость 2: скользящее окно фиксированной длины (те же гиперпараметры)
    log(f"скользящее окно {ROLLING_WINDOW}")
    roll_models = [(Drift(), 1), (ARIMA(), 1), (ETS(), 1),
                   (LightGBM(params=info["LightGBM"]), 5),
                   (GRU(lookback=info["GRU"]["lookback"], hidden=info["GRU"]["hidden"]), 20)]
    rp = [pred[pred.model == "Naive"]]
    for m, k in roll_models:
        rp.append(walk_forward(m, df, origins, refit_every=k, window=ROLLING_WINDOW)[0])
    roll = metric_table(pd.concat(rp, ignore_index=True))
    roll["window"] = f"rolling_{ROLLING_WINDOW}"
    exp = mt[["model", "h", "MAE", "RMSE", "sMAPE", "DirAcc", "n"]].assign(window="expanding")
    pd.concat([exp, roll]).to_csv(RESULTS / "robust_window.csv", index=False, float_format="%.4f")

    # --- устойчивость 3: seed нейросети
    mem = pd.concat(members, ignore_index=True)
    mem_mt = metric_table(mem, by=("model", "member", "h"))
    mem_mt.to_csv(RESULTS / "robust_gru_seeds.csv", index=False, float_format="%.4f")

    plot_forecasts(pred, df, origins)
    plot_metrics(mt)
    plot_cum_error(pred)
    plot_seeds(mem_mt, mt)
    log("Готово. Результаты в results/")


if __name__ == "__main__":
    main()
