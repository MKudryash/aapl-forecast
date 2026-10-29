"""Загрузка, проверка и очистка дневных котировок Dukascopy.

Поддерживаются два формата экспорта:
1) dukascopy-node:  timestamp(ms),open,high,low,close[,volume]
2) сайт Dukascopy:  "Gmt time","Open","High","Low","Close","Volume"
   (дата вида 01.10.2025 00:00:00.000)

Принцип: то, что можно исправить однозначно (выходные, праздничные "плоские" свечи),
удаляется с записью в отчет. То, что исправить однозначно нельзя (дубликаты дат,
неверный порядок OHLC, неполный период, неожиданные пропуски), останавливает расчет:
такие данные нужно сверить с источником вручную, а не "чинить" молча.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from config import DATA_RAW, DATA_CLEAN, RESULTS, PERIOD_START, PERIOD_END, EXCHANGE_HOLIDAYS


class DataError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


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
        "open": pd.to_numeric(df["open"], errors="raise"),
        "high": pd.to_numeric(df["high"], errors="raise"),
        "low": pd.to_numeric(df["low"], errors="raise"),
        "close": pd.to_numeric(df["close"], errors="raise"),
    })
    out["volume"] = pd.to_numeric(df["volume"], errors="raise") if "volume" in df.columns else np.nan
    return out


def clean(df, strict=True):
    """Возвращает очищенный ряд торговых дней и отчет об очистке."""
    rep = {"rows_raw": int(len(df))}

    # 1. Дубликаты дат: какую из двух свечей оставить, однозначно решить нельзя
    if df["date"].duplicated().any():
        raise DataError(f"Дубликаты дат: {df.loc[df.date.duplicated(), 'date'].dt.date.tolist()}")
    df = df.sort_values("date")

    # 2. Только период задания
    inside = df["date"].between(PERIOD_START, PERIOD_END)
    rep["outside_period_removed"] = int((~inside).sum())
    df = df[inside]

    # 3. Выходные
    wknd = df["date"].dt.dayofweek >= 5
    rep["weekend_rows_removed"] = int(wknd.sum())
    df = df[~wknd]

    # 4. Плоские свечи неторговых дней: O=H=L=C без объема (повтор прошлого закрытия)
    flat = df["high"] == df["low"]
    if df["volume"].notna().any():
        flat = flat & (df["volume"].fillna(0) == 0)
    rep["flat_nontrading_removed"] = int(flat.sum())
    rep["flat_nontrading_dates"] = df.loc[flat, "date"].dt.strftime("%Y-%m-%d").tolist()
    df = df[~flat].reset_index(drop=True)

    # 5. Корректность значений: конечные, положительные, low <= open,close <= high
    px = df[["open", "high", "low", "close"]]
    if not np.isfinite(px.values).all():
        raise DataError("Есть пустые или бесконечные цены")
    ok = (df.low > 0) & (df.low <= px[["open", "close"]].min(axis=1)) & \
         (df.high >= px[["open", "close"]].max(axis=1))
    if not ok.all():
        raise DataError(f"Нарушен порядок OHLC в датах: {df.loc[~ok, 'date'].dt.date.tolist()}")
    rep["ohlc_checks_passed"] = True

    # 6. Полнота периода
    first, last = df.date.iloc[0].strftime("%Y-%m-%d"), df.date.iloc[-1].strftime("%Y-%m-%d")
    rep["period"] = [first, last]
    if strict and (first != PERIOD_START or last != PERIOD_END):
        raise DataError(f"Период {first}..{last} не совпадает с заданием {PERIOD_START}..{PERIOD_END}")

    # 7. Пропуски будних дней сверяются с календарем праздников биржи
    bdays = pd.bdate_range(df.date.min(), df.date.max())
    missing = sorted(d.strftime("%Y-%m-%d") for d in set(bdays) - set(df.date))
    unexpected = sorted(set(missing) - set(EXCHANGE_HOLIDAYS))
    rep["missing_weekdays"] = missing
    rep["missing_weekdays_are_exchange_holidays"] = not unexpected
    if strict and unexpected:
        raise DataError(f"Пропущены будни, не являющиеся праздниками биржи: {unexpected}")

    rep["rows_clean"] = int(len(df))
    return df, rep


def main(raw_path=None):
    if raw_path is None:
        files = sorted(DATA_RAW.glob("*.csv"))
        if not files:
            sys.exit(f"Положите CSV Dukascopy в {DATA_RAW}")
        raw_path = files[0]
    df, rep = clean(load_raw(raw_path))
    rep["source_file"] = str(raw_path.name if hasattr(raw_path, "name") else raw_path)
    rep["source_sha256"] = sha256(raw_path)
    df.to_csv(DATA_CLEAN, index=False, float_format="%.4f")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "cleaning_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("flat_nontrading_dates", "missing_weekdays")},
                     ensure_ascii=False))
    return df


def load_clean():
    return pd.read_csv(DATA_CLEAN, parse_dates=["date"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
