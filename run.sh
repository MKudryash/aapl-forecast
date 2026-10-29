#!/bin/bash
# Полный запуск проекта на macOS/Linux: окружение -> зависимости -> эксперимент.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo ">> Создаю виртуальное окружение .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate
echo ">> Устанавливаю зависимости (первый раз несколько минут)"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if ! ls data/raw/*.csv >/dev/null 2>&1; then
  echo "!! В папке data/raw нет CSV с котировками AAPL."
  echo "   Выгрузите его с Dukascopy и положите в data/raw/, затем запустите ./run.sh снова."
  exit 1
fi

echo ">> Запускаю эксперимент"
python src/run_all.py   # ~4 минуты, в конце проверка на утечку
echo ">> Готово: таблицы в results/, графики в results/figures/"
