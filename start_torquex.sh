#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then python3.12 -m venv .venv; fi
if ! cmp -s requirements.txt .venv/installed_requirements.txt; then
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/installed_requirements.txt
fi
exec .venv/bin/python -m streamlit run app/app.py
