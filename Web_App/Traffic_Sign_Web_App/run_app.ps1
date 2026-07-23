$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LocalPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ParentPython = Join-Path (Split-Path $PSScriptRoot -Parent) ".venv\Scripts\python.exe"

if (Test-Path $LocalPython) {
    & $LocalPython -m streamlit run app.py
}
elseif (Test-Path $ParentPython) {
    & $ParentPython -m streamlit run app.py
}
else {
    python -m streamlit run app.py
}
