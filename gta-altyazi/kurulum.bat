@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul || (echo Python bulunamadi. PowerShell'de:  winget install Python.Python.3.12 & pause & exit /b 1)
rem Bagimliliklarin hazir paketleri 3.11-3.13 icin var; daha yeni Python derleyici ister.
set PYV=
for %%v in (3.12 3.13 3.11) do (
  if not defined PYV (
    py -%%v -c "pass" >nul 2>nul && set PYV=%%v
  )
)
if not defined PYV (
  echo Python 3.12 gerekli ^(yuklu surumler cok yeni ya da eski^). PowerShell'de:
  echo     winget install Python.Python.3.12
  echo sonra kurulum.bat'i tekrar calistirin. Diger Python surumleri kalabilir.
  pause & exit /b 1
)
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" || (
    echo Eski sanal ortam uygun olmayan Python surumuyle kurulmus; yeniden kuruluyor...
    rmdir /s /q .venv
  )
)
if not exist .venv\Scripts\python.exe (py -%PYV% -m venv .venv || (pause & exit /b 1))
echo Python %PYV% kullaniliyor.
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-gpu.txt || (pause & exit /b 1)
rem OCR ekran kartinda (DirectML) calissin: onnxruntime yerine onnxruntime-directml (yoksa kur).
.venv\Scripts\python -c "import onnxruntime as o, sys; sys.exit('DmlExecutionProvider' not in o.get_available_providers())" 2>nul || .venv\Scripts\python -m pip install -q --no-deps --force-reinstall "onnxruntime-directml>=1.17,<2" || echo DirectML kurulamadi; OCR islemcide calisacak.
echo.
if not exist model\model.bin (
  echo UYARI: Yerel taslak ceviri modeli yok. Indirdiginiz GTA-Altyazi exe klasorundeki "model" klasorunu
  echo buraya ^(%~dp0^) kopyalayin. Model olmadan da calisir, ama yalnizca son cevirmen kullanilir.
  echo.
)
echo Kurulum tamam. baslat.bat ile calistirin. Guncellemek icin guncelle.bat.
pause
