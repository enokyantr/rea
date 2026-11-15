@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul || (echo Python bulunamadi. https://www.python.org/downloads/ adresinden Python 3.12 kurun ^(kurulumda "Add python.exe to PATH" isaretleyin^). & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (py -3 -m venv .venv || (pause & exit /b 1))
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-gpu.txt || (pause & exit /b 1)
echo.
if not exist model\model.bin (
  echo UYARI: Yerel taslak ceviri modeli yok. Indirdiginiz GTA-Altyazi exe klasorundeki "model" klasorunu
  echo buraya ^(%~dp0^) kopyalayin. Model olmadan da calisir, ama yalnizca son cevirmen kullanilir.
  echo.
)
echo Kurulum tamam. baslat.bat ile calistirin. Guncellemek icin guncelle.bat.
pause
