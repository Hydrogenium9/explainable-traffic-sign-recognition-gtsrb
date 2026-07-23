@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py
    exit /b %errorlevel%
)

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -m streamlit run app.py
    exit /b %errorlevel%
)

python -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo Streamlit could not start.
    echo Run setup_app.bat first or activate a Python environment with the requirements installed.
    pause
)
