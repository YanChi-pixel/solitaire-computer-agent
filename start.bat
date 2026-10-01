@echo off
rem ============================================================
rem  Solitaire AI - one-click launch.
rem  - kills any leftover server on port 8000
rem  - starts the server, waits until it is actually up
rem  - opens the browser at 127.0.0.1 (NOT "localhost" — the server
rem    binds IPv4 only, and "localhost" can resolve to ::1 and fail)
rem ============================================================

cd /d "%~dp0"

echo [1/3] Cleaning port 8000 (killing any old server)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo   killing PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo [2/3] Starting server...
start "Solitaire AI server" /b python server.py

echo   waiting for the server to come up...
set /a tries=0
:waitloop
timeout /t 1 /nobreak >nul
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel%==0 goto ready
set /a tries+=1
if %tries% geq 60 goto timeout
goto waitloop

:ready
echo [3/3] Server is up. Opening browser at http://127.0.0.1:8000
start http://127.0.0.1:8000
echo.
echo Server is running. Stop with the "Stop server" button in the page,
echo or close this window (or Ctrl+C).
exit /b 0

:timeout
echo [ERROR] Server did not start within 60 seconds.
echo         Run  python server.py  directly to see the error output.
pause
exit /b 1
