@echo off
title GTA Altyazi
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Once kurulum.bat calistirin. & pause & exit /b 1)
:calistir
.venv\Scripts\python -m altyazi %*
rem 3: menuden "Kaydet ve yeniden baslat"
if errorlevel 3 if not errorlevel 4 goto calistir
if errorlevel 1 pause
