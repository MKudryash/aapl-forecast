"""Единые настройки эксперимента. Меняются только по общему согласию группы."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "aapl_d1_clean.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

INSTRUMENT = "AAPL.US/USD (CFD, Dukascopy)"
TIMEZONE = "UTC (GMT)"

# Период задания: 12 месяцев на дату выдачи 28.10.2026 (сам день выдачи не входит)
PERIOD_START = "2025-10-28"
PERIOD_END = "2026-10-27"

# Праздники бирж Nasdaq/NYSE в периоде (дни без торгов в будни).
# Источники: https://www.nasdaqtrader.com/trader.aspx?id=calendar
#            https://www.nyse.com/markets/hours-calendars
EXCHANGE_HOLIDAYS = [
    "2025-11-27",  # День благодарения
    "2025-12-25",  # Рождество
    "2026-01-01",  # Новый год
    "2026-01-19",  # День Мартина Лютера Кинга
    "2026-02-16",  # День президентов
    "2026-04-03",  # Страстная пятница
    "2026-05-25",  # День памяти
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # День независимости (перенос с субботы 04.07)
    "2026-09-07",  # День труда
]

HORIZONS = [1, 5]          # горизонты прогноза, торговых дней
H_MAX = max(HORIZONS)
TEST_SIZE = 60             # число точек прогноза (origins) в тесте, ~3 месяца
VAL_SIZE = 40              # хвост обучающей части для подбора гиперпараметров
ROLLING_WINDOW = 150       # длина окна для проверки устойчивости (скользящее окно)

# Как часто переобучать модель (в точках прогноза). 1 = каждый торговый день.
# GRU -- раз в неделю (5 торговых дней): ежедневно ансамбль из 5 сетей обучался бы ~15 минут.
REFIT_EVERY = {"Naive": 1, "Drift": 1, "ARIMA": 1, "ETS": 1, "LightGBM": 1, "GRU": 5}

SEED = 42
NN_SEEDS = [0, 1, 2, 3, 4]  # ансамбль нейросети и анализ устойчивости к seed

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_BLOCKS = [1, 5, 10]   # длина блока; основная -- 5 (= максимальный горизонт)
