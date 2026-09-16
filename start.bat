@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo    Inference Quality Test (Python, standalone)
echo ============================================================
echo.
echo [1/3] Checking Python...
python --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Python not found. Please install Python 3.9+ from:
  echo         https://www.python.org/downloads/
  echo         IMPORTANT: check "Add python.exe to PATH" when installing.
  echo.
  pause
  exit /b 1
)
echo [2/3] Preparing virtual environment + dependencies (first run may take a while)...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install -q -r requirements.txt
echo [3/3] Starting server, browser will open automatically...
start "" http://127.0.0.1:17889/
python server.py
echo.
echo Server stopped.
pause
