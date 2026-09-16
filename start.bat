@echo off
rem ============================================================
rem  Inference Quality Test (Python, standalone) - Windows launcher
rem  NOTE: keep this file ASCII + CRLF, cmd.exe parses it strictly.
rem ============================================================
setlocal
cd /d "%~dp0"
echo ============================================================
echo    Inference Quality Test (Python, standalone)
echo ============================================================
echo.
echo [1/4] Checking Python...
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
echo [2/4] Preparing virtual environment + dependencies (first run may take a while)...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install -q -r requirements.txt

rem ---------------------------------------------------------------
rem [3/4] Port self-healing.
rem   If the port is already taken (typical when a previous server did
rem   not exit cleanly) the owning process is terminated, the port is
rem   checked again, and this repeats until it is free -> avoids the
rem   "[Errno 10048] only one usage of each socket address" startup error.
rem   Override the port with QTEST_PORT (same variable server.py reads).
rem ---------------------------------------------------------------
if not defined QTEST_PORT set "QTEST_PORT=17889"
set "PORT=%QTEST_PORT%"
set /a TRIES=0
echo [3/4] Checking whether port %PORT% is free...

:PORTCHK
set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do call :KILLPID %%P
if not defined FOUND goto PORTOK
set /a TRIES+=1
if %TRIES% GTR 15 goto PORTFAIL
ping -n 2 127.0.0.1 >nul
goto PORTCHK

:PORTOK
echo    [port] port %PORT% is free. (checked %TRIES% time(s))
echo [4/4] Starting server, browser will open automatically...
start "" http://127.0.0.1:%PORT%/
python server.py
echo.
echo Server stopped.
pause
exit /b 0

:KILLPID
rem %1 = PID listening on the port
set "FOUND=1"
if "%1"=="0" (
  echo    [port] WARNING: port %PORT% is held by System PID 0, cannot terminate.
  goto :eof
)
if "%1"=="4" (
  echo    [port] WARNING: port %PORT% is held by System PID 4, cannot terminate.
  goto :eof
)
set "PNAME=unknown"
for /f "tokens=1 delims=," %%N in ('tasklist /FI "PID eq %1" /NH /FO CSV 2^>nul') do set "PNAME=%%~N"
echo    [port] port %PORT% is in use by PID %1 (%PNAME%) - terminating it...
taskkill /F /T /PID %1 >nul 2>nul
if errorlevel 1 echo    [port] WARNING: failed to terminate PID %1 (try running as Administrator).
goto :eof

:PORTFAIL
echo.
echo    [port] ERROR: port %PORT% is still in use after %TRIES% attempts.
echo           Find and close the program below manually, then re-run start.bat:
echo             netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"
echo             tasklist /FI "PID eq ^<PID^>"
echo             taskkill /F /PID ^<PID^>
echo           Or start on another port:
echo             set QTEST_PORT=18000
echo             start.bat
echo.
pause
exit /b 1
