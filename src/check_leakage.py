"""Проверка отсутствия утечки будущего (заглядывания вперед).

Идея: если прогноз в точке t использует только данные до t, то при замене ВСЕХ строк
после t на случайный мусор прогноз в точке t не должен измениться ни на цент.
Проверка выполняется для каждой модели в трех точках теста (начало, середина, конец),
плюс отдельно проверяется, что признаки строки t не зависят от будущих строк.

Запуск: python src/check_leakage.py   (также вызывается из run_all.py)
"""
import json

import numpy as np
import pandas as pd

import data as data_mod
from config import RESULTS
from evaluation import walk_forward, test_origins
from features import make_features
from models import Naive, Drift, ARIMA, ETS, LightGBM, GRU


def corrupt_future(df, t, seed=0):
    """Копия df, где все строки после t заменены случайными ценами (порядок OHLC сохранен)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(df) - t - 1
    base = df["close"].iloc[t] * rng.uniform(0.5, 1.5, n)
    o, c = base * rng.uniform(0.97, 1.03, n), base * rng.uniform(0.97, 1.03, n)
    out.loc[t + 1:, "open"], out.loc[t + 1:, "close"] = o, c
    out.loc[t + 1:, "high"] = np.maximum(o, c) * 1.01
    out.loc[t + 1:, "low"] = np.minimum(o, c) * 0.99
    return out


def run(df=None, params=None):
    df = data_mod.load_clean() if df is None else df
    params = params or {}
    origins = test_origins(len(df))
    checks = [int(origins[0]), int(origins[len(origins) // 2]), int(origins[-1])]
    res = {"origins_checked": [str(df.date.iloc[t].date()) for t in checks], "models": {}}

    # 1. Признаки
    ok = True
    for t in checks:
        f1 = make_features(df).iloc[:t + 1]
        f2 = make_features(corrupt_future(df, t)).iloc[:t + 1]
        ok &= f1.equals(f2)
    res["features_independent_of_future"] = bool(ok)

    # 2. Прогнозы каждой модели
    def factories():
        lg = params.get("LightGBM")
        gr = params.get("GRU", {})
        return [Naive, Drift, ARIMA, ETS,
                lambda: LightGBM(params=lg),
                lambda: GRU(lookback=gr.get("lookback"), hidden=gr.get("hidden"), seeds=[0])]

    for make in factories():
        diffs = []
        for t in checks:
            p1 = walk_forward(make(), df, [t])[0]
            p2 = walk_forward(make(), corrupt_future(df, t), [t])[0]
            diffs.append(float(np.max(np.abs(p1.y_pred.values - p2.y_pred.values))))
        name = make().name
        res["models"][name] = {"max_abs_diff_usd": max(diffs), "passed": max(diffs) < 1e-9}
        print(f"  {name:9s} max |Δпрогноза| = {max(diffs):.2e}  {'OK' if max(diffs) < 1e-9 else 'УТЕЧКА!'}")
    res["all_passed"] = res["features_independent_of_future"] and all(v["passed"] for v in res["models"].values())
    (RESULTS / "leakage_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    if not res["all_passed"]:
        raise AssertionError("Обнаружена зависимость прогноза от будущих данных, см. results/leakage_check.json")
    return res


if __name__ == "__main__":
    params = None
    hp = RESULTS / "chosen_hyperparams.json"
    if hp.exists():
        params = json.loads(hp.read_text())
    print(json.dumps(run(params=params), ensure_ascii=False, indent=2))
