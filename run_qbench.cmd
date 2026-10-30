@echo off
rem Q-Bench runner: double-click to start. Closing this window stops the run.
setlocal
cd /d "%~dp0"
set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%GITBASH%" (
  echo Git Bash not found: %GITBASH%
  pause
  exit /b 1
)
"%GITBASH%" ./run_qbench.sh
echo.
echo Runner exited with code %ERRORLEVEL%.
pause
