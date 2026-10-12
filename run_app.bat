@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=..\..\p12\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment was not found: %PYTHON%
  echo Expected location: C:\Users\bakon\sabak\p12
  exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8501"

echo Starting AI Security Monitor at http://127.0.0.1:%PORT%
"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1 --server.port %PORT%

