$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project"
$CodeRoot = Join-Path $ProjectRoot "Codes\Data_Preparation"
$VenvRoot = Join-Path $ProjectRoot ".venv"

Set-Location $CodeRoot

if (-not (Test-Path $VenvRoot)) {
    Write-Host "Создание виртуального окружения..."
    py -3.11 -m venv $VenvRoot
}

$Python = Join-Path $VenvRoot "Scripts\python.exe"

Write-Host "Обновление pip..."
& $Python -m pip install --upgrade pip

Write-Host "Установка библиотек..."
& $Python -m pip install -r requirements.txt

Write-Host "Запуск подготовки данных..."
& $Python prepare_and_verify_data.py
