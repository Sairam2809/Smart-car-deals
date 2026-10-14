@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating the TORQUEX Python environment...
    py -3.12 -m venv .venv >nul 2>&1
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 (
            echo Please install Python, then run this file again.
            pause
            exit /b 1
        )
    )
)
fc /b requirements.txt .venv\installed_requirements.txt >nul 2>&1
if errorlevel 1 (
    echo First-time setup: downloading dependencies. This can take several minutes.
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installation failed. Check the message above and your internet connection.
        pause
        exit /b 1
    )
    copy /y requirements.txt .venv\installed_requirements.txt >nul
)
echo Opening TORQUEX. Keep this window open while using the app.
.venv\Scripts\python.exe -m streamlit run app/app.py
pause
