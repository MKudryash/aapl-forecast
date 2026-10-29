"""Walk-forward валидация, метрики, тест Диболда-Мариано и блочный bootstrap."""
import time

import numpy as np
import pandas as pd
from scipy import stats

from config import HORIZONS, H_MAX, TEST_SIZE, BOOTSTRAP_REPEATS, BOOTSTRAP_BLOCKS


def test_origins(n, test_size=TEST_SIZE):
    """Индексы точек прогноза t. Одинаковы для всех моделей и горизонтов:
    для каждой t известны факты и на t+1, и на t+5."""
    last = n - 1 - H_MAX
    return np.arange(last - test_size + 1, last + 1)


def walk_forward(model, df, origins, refit_every=1, window=None):
    """В каждой точке t модель получает только df[:t+1] (или последние window строк),
    переобучается раз в refit_every точек и прогнозирует t+1..t+H_MAX."""
    rows, members = [], []
    fit_time = pred_time = 0.0
    for i, t in enumerate(origins):
        start = 0 if window is None else max(0, t + 1 - window)
        hist = df.iloc[start:t + 1].reset_index(drop=True)
        if i % refit_every == 0:
            t0 = time.perf_counter()
            model.fit(hist)
            fit_time += time.perf_counter() - t0
        t0 = time.perf_counter()
        path = model.predict(hist)
        pred_time += time.perf_counter() - t0
        mem = getattr(model, "last_members", None)
        for h in HORIZONS:
            rows.append(dict(model=model.name, origin=df["date"].iloc[t], target=df["date"].iloc[t + h],
                             h=h, y_last=df["close"].iloc[t], y_true=df["close"].iloc[t + h],
                             y_pred=float(path[h - 1])))
            if mem is not None:
                for s, m in enumerate(mem):
                    members.append(dict(model=model.name, member=s, origin=df["date"].iloc[t], h=h,
                                        y_last=df["close"].iloc[t], y_true=df["close"].iloc[t + h],
                                        y_pred=float(m[h - 1])))
    timing = dict(model=model.name, fit_s=fit_time, predict_s=pred_time,
                  n_fits=int(np.ceil(len(origins) / refit_every)))
    return pd.DataFrame(rows), timing, pd.DataFrame(members)


def metrics(g):
    e = g["y_pred"] - g["y_true"]
    smape = 100 * np.mean(2 * np.abs(e) / (np.abs(g["y_true"]) + np.abs(g["y_pred"])))
    # Изменения меньше 1e-5 от цены считаем нулевыми: это шум округления
    # (например, у ARIMA(0,1,0) при переходе от логарифма обратно к цене)
    diff = g["y_pred"] - g["y_last"]
    d_pred = np.sign(diff.where(diff.abs() > 1e-5 * g["y_last"], 0.0))
    d_true = np.sign(g["y_true"] - g["y_last"])
    # У наивного прогноза изменение всегда 0: направление не определено
    da = np.nan if (d_pred == 0).all() else 100 * np.mean(d_pred == d_true)
    return pd.Series(dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                          sMAPE=smape, DirAcc=da, n=len(g)))


def metric_table(pred, by=("model", "h")):
    return pred.groupby(list(by)).apply(metrics, include_groups=False).reset_index()


def dm_test(e1, e2, h, power=2):
    """Тест Диболда-Мариано с поправкой Harvey-Leybourne-Newbold.
    H0: равная точность. stat<0 -- модель 1 точнее модели 2."""
    if np.allclose(e1, e2, rtol=0, atol=1e-6):
        return np.nan, np.nan          # прогнозы совпадают (например, ARIMA(0,1,0) и Naive)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return stat, p


def dm_vs_baseline(pred, baseline="Naive"):
    rows = []
    for h in HORIZONS:
        ref = pred[(pred.model == baseline) & (pred.h == h)].sort_values("origin")
        e_ref = (ref.y_pred - ref.y_true).values
        for m in pred.model.unique():
            if m == baseline:
                continue
            g = pred[(pred.model == m) & (pred.h == h)].sort_values("origin")
            e = (g.y_pred - g.y_true).values
            for power, loss in ((2, "MSE"), (1, "MAE")):
                s, p = dm_test(e, e_ref, h, power)
                rows.append(dict(model=m, h=h, loss=loss, DM=s, p_value=p))
    return pd.DataFrame(rows)


def block_bootstrap_ci(d, block=5, repeats=BOOTSTRAP_REPEATS, seed=2026, level=0.95):
    """Круговой блочный bootstrap (circular moving block) для среднего ряда d.
    Ряд нарезается на блоки подряд идущих значений длиной block, блоки выбираются
    с возвращением (с "заворотом" через конец ряда), из них собирается новый ряд той же длины.
    Блоки сохраняют автокорреляцию соседних ошибок (перекрывающиеся 5-дневные прогнозы)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    n = len(d)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(repeats, k))
    idx = (starts[:, :, None] + np.arange(block)) % n
    means = d[idx.reshape(repeats, -1)[:, :n]].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def bootstrap_vs_baseline(pred, baseline="Naive", blocks=BOOTSTRAP_BLOCKS):
    """95% интервал для ΔMAE = MAE(модель) - MAE(baseline).
    Интервал целиком ниже 0 -- модель точнее; целиком выше 0 -- хуже; содержит 0 -- различие не доказано."""
    rows = []
    for h in HORIZONS:
        ref = pred[(pred.model == baseline) & (pred.h == h)].sort_values("origin")
        ae_ref = (ref.y_pred - ref.y_true).abs().values
        for m in pred.model.unique():
            if m == baseline:
                continue
            g = pred[(pred.model == m) & (pred.h == h)].sort_values("origin")
            d = (g.y_pred - g.y_true).abs().values - ae_ref
            for b in blocks:
                if np.allclose(d, 0, atol=1e-6):
                    lo = hi = 0.0
                else:
                    lo, hi = block_bootstrap_ci(d, block=b)
                verdict = "лучше" if hi < 0 else ("хуже" if lo > 0 else "не различимо")
                rows.append(dict(model=m, h=h, block=b, delta_MAE=float(d.mean()),
                                 ci_low=lo, ci_high=hi, verdict=verdict))
    return pd.DataFrame(rows)
