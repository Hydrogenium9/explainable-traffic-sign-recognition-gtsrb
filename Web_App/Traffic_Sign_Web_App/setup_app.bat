@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher "py" was not found.
    echo Install 64-bit Python 3.13 and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment...
    py -3.13 -m venv .venv
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

python validate_installation.py

if errorlevel 1 (
    echo Validation failed.
    pause
    exit /b 1
)

echo Setup completed successfully.
pause
