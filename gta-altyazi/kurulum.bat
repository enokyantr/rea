@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul || (echo Python bulunamadi. https://www.python.org/downloads/ adresinden Python 3.12+ kurun ^(kurulumda "Add to PATH" isaretleyin^). & pause & exit /b 1)
py -3 -m venv .venv || (pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
echo.
echo Kurulum tamam. config.json icine DeepL anahtarinizi yazip baslat.bat ile calistirin.
pause
