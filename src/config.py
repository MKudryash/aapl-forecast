"""Единые настройки эксперимента. Меняются только по общему согласию группы."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "aapl_d1_clean.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

INSTRUMENT = "AAPL.US/USD (CFD, Dukascopy)"
TIMEZONE = "UTC (GMT)"

HORIZONS = [1, 5]          # горизонты прогноза, торговых дней
H_MAX = max(HORIZONS)
TEST_SIZE = 60             # число точек прогноза (origins) в тесте, ~3 месяца
VAL_SIZE = 40              # хвост обучающей части для подбора гиперпараметров
ROLLING_WINDOW = 150       # длина окна для проверки устойчивости (скользящее окно)

SEED = 42
NN_SEEDS = [0, 1, 2, 3, 4]  # для анализа устойчивости нейросети
