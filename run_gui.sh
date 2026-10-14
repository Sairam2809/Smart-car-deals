#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1
python -m streamlit run app/app.py
