"""Загрузка и очистка дневных котировок Dukascopy.

Поддерживаются два формата экспорта:
1) dukascopy-node:  timestamp(ms),open,high,low,close[,volume]
2) сайт Dukascopy:  "Gmt time","Open","High","Low","Close","Volume"
   (дата вида 01.10.2025 00:00:00.000)
"""
import json
import sys

import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_CLEAN, RESULTS


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if np.issubdtype(ts.dtype, np.number):
            date = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            date = pd.to_datetime(ts, utc=True)
    elif "gmt time" in df.columns:
        date = pd.to_datetime(df["gmt time"], format="%d.%m.%Y %H:%M:%S.%f", utc=True)
    else:
        col = [c for c in df.columns if "date" in c or "time" in c][0]
        date = pd.to_datetime(df[col], utc=True, dayfirst=True)
    out = pd.DataFrame({
        "date": date.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
    })
    out["volume"] = df["volume"].astype(float) if "volume" in df.columns else np.nan
    return out


def clean(df):
    """Возвращает очищенный ряд торговых дней и отчет об очистке."""
    rep = {"rows_raw": int(len(df))}
    df = df.sort_values("date")
    dup = df.duplicated("date", keep="last")
    rep["duplicates_removed"] = int(dup.sum())
    df = df[~dup]

    wknd = df["date"].dt.dayofweek >= 5
    rep["weekend_rows_removed"] = int(wknd.sum())
    df = df[~wknd]

    # Неторговые дни (праздники NYSE): Dukascopy отдает "плоскую" свечу
    # O=H=L=C без объема, равную предыдущему закрытию.
    flat = (df["high"] == df["low"])
    if df["volume"].notna().any():
        flat = flat & (df["volume"].fillna(0) == 0)
    rep["flat_nontrading_removed"] = int(flat.sum())
    rep["flat_nontrading_dates"] = df.loc[flat, "date"].dt.strftime("%Y-%m-%d").tolist()
    df = df[~flat]

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | df["close"].isna()
    rep["invalid_rows_removed"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)

    # Пропуски в рабочих днях (кроме праздников) после очистки
    bdays = pd.bdate_range(df["date"].min(), df["date"].max())
    missing = sorted(set(bdays) - set(df["date"]))
    rep["missing_weekdays"] = [d.strftime("%Y-%m-%d") for d in missing]
    rep["rows_clean"] = int(len(df))
    rep["period"] = [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")]
    return df, rep


def main(raw_path=None):
    if raw_path is None:
        files = sorted(DATA_RAW.glob("*.csv"))
        if not files:
            sys.exit(f"Положите CSV Dukascopy в {DATA_RAW}")
        raw_path = files[0]
    df, rep = clean(load_raw(raw_path))
    rep["source_file"] = str(raw_path.name if hasattr(raw_path, "name") else raw_path)
    df.to_csv(DATA_CLEAN, index=False, float_format="%.4f")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cleaning_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "flat_nontrading_dates"}, ensure_ascii=False))
    return df


def load_clean():
    return pd.read_csv(DATA_CLEAN, parse_dates=["date"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
