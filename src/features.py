"""Признаки для ML-модели. Все признаки в строке t используют только данные до t включительно."""
import numpy as np
import pandas as pd

N_LAGS = 10


def make_features(df):
    c = np.log(df["close"])
    r = c.diff()
    f = pd.DataFrame(index=df.index)
    for k in range(N_LAGS):
        f[f"r_lag{k}"] = r.shift(k)
    for w in (5, 10, 20):
        f[f"r_mean{w}"] = r.rolling(w).mean()
        f[f"r_std{w}"] = r.rolling(w).std()
    f["dist_ma20"] = c - c.rolling(20).mean()
    hl = np.log(df["high"] / df["low"])
    for k in range(3):
        f[f"hl_lag{k}"] = hl.shift(k)
    f["oc"] = np.log(df["close"] / df["open"])
    f["dow"] = df["date"].dt.dayofweek
    return f


def make_targets(df, horizons):
    """cumret_h[t] = log(close[t+h] / close[t])."""
    c = np.log(df["close"])
    return pd.DataFrame({h: c.shift(-h) - c for h in horizons}, index=df.index)
