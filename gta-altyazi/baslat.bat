@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Once kurulum.bat calistirin. & pause & exit /b 1)
.venv\Scripts\python -m altyazi %*
if errorlevel 1 pause
