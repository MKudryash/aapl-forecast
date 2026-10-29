"""Автоматическая сводка results/REPORT.md. Все числа берутся из файлов results/,
поэтому сводка всегда соответствует последнему запуску (в отличие от README, где цифры
вписаны вместе с интерпретацией и должны обновляться вручную при смене данных)."""
import json

import numpy as np
import pandas as pd

from config import RESULTS, HORIZONS


def md_table(df, floatfmt="{:.3f}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def write_report():
    R = RESULTS
    meta = json.loads((R / "metadata.json").read_text())
    eda = json.loads((R / "eda_stats.json").read_text())
    leak = json.loads((R / "leakage_check.json").read_text())
    hp = json.loads((R / "chosen_hyperparams.json").read_text())
    mt = pd.read_csv(R / "metrics.csv")
    dm = pd.read_csv(R / "dm_tests.csv")
    bt = pd.read_csv(R / "bootstrap_ci.csv")
    up = pd.read_csv(R / "direction_reference.csv")
    tm = pd.read_csv(R / "timing.csv")

    L = ["# Автоматическая сводка результатов", "",
         f"Файл данных: `{meta['source_file']}`, SHA-256 `{meta['source_sha256']}`.",
         f"Наблюдений: {meta['rows']} ({meta['period'][0]} - {meta['period'][1]}). "
         f"Точек прогноза в тесте: {meta['test_origins']} ({meta['first_origin']} - {meta['last_origin']}).",
         f"Python {meta['python']}, {meta['platform']}.", "",
         f"Выбранные гиперпараметры: ARIMA{tuple(hp['ARIMA']['order'])} trend={hp['ARIMA']['trend']}; "
         f"LightGBM {hp['LightGBM']}; GRU lookback={hp['GRU']['lookback']}, hidden={hp['GRU']['hidden']}.", "",
         f"Анализ ряда: ADF p(лог-цена) = {eda['adf_p_logprice']:.3f}, ADF p(доходности) = {eda['adf_p_returns']:.1e}, "
         f"Льюнг-Бокс p(лаг 10) = {eda['ljungbox_p_lag10']:.3f}.", ""]
    for h in HORIZONS:
        t = mt[mt.h == h][["model", "MAE", "RMSE", "sMAPE", "DirAcc", "MAE_vs_Naive"]].sort_values("MAE")
        share = float(up.loc[up.h == h, "share_up_pct"].iloc[0])
        L += [f"## Горизонт h = {h}", "", md_table(t), "",
              f"Ориентир для DirAcc (доля роста в тесте, стратегия «всегда рост»): {share:.1f}%.", ""]
    d = dm.pivot_table(index="model", columns=["h", "loss"], values=["DM", "p_value"])
    rows = []
    for m in d.index:
        rows.append([m] + [f"{d.loc[m, ('DM', h, l)]:+.2f} (p={d.loc[m, ('p_value', h, l)]:.3f})"
                           if pd.notna(d.loc[m, ('DM', h, l)]) else "совпадает с Naive"
                           for h in HORIZONS for l in ("MSE", "MAE")])
    L += ["## Тест Диболда-Мариано против Naive", "", "DM < 0 — модель точнее; p < 0.05 — различие значимо.", "",
          md_table(pd.DataFrame(rows, columns=["model"] + [f"h={h}, {l}" for h in HORIZONS for l in ("MSE", "MAE")])), ""]
    b = bt[bt.block == 5][["model", "h", "delta_MAE", "ci_low", "ci_high", "verdict"]]
    L += ["## Блочный bootstrap: 95% интервал ΔMAE = MAE(модель) - MAE(Naive), блок 5", "",
          md_table(b), "", "Чувствительность к длине блока (1, 5, 10): `bootstrap_ci.csv`.", ""]
    L += ["## Вычислительные затраты, секунды", "", md_table(tm, "{:.3f}"), ""]
    L += ["## Проверка на утечку будущего", "",
          f"Строки после точки прогноза заменены случайными ценами, точки: {', '.join(leak['origins_checked'])}.",
          f"Признаки не зависят от будущего: {'да' if leak['features_independent_of_future'] else 'НЕТ'}.", ""]
    L += [f"- {m}: максимальное изменение прогноза {v['max_abs_diff_usd']:.1e} USD — "
          f"{'пройдено' if v['passed'] else 'НЕ пройдено'}" for m, v in leak["models"].items()]
    (R / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
