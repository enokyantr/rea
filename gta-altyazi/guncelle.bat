@echo off
chcp 65001 >nul
cd /d "%~dp0"
where git >nul 2>nul || (echo Git bulunamadi. https://git-scm.com/download/win adresinden kurun. & pause & exit /b 1)
echo Son surum GitHub'dan aliniyor...
git pull --ff-only || (echo Guncelleme basarisiz. Yerel degisiklik varsa bildirin. & pause & exit /b 1)
.venv\Scripts\python -m pip install -q -r requirements.txt -r requirements-gpu.txt || (pause & exit /b 1)
rem OCR ekran kartinda (DirectML) calissin: onnxruntime yerine onnxruntime-directml (yoksa kur).
.venv\Scripts\python -c "import onnxruntime as o, sys; sys.exit('DmlExecutionProvider' not in o.get_available_providers())" 2>nul || .venv\Scripts\python -m pip install -q --no-deps --force-reinstall "onnxruntime-directml>=1.17,<2" || echo DirectML kurulamadi; OCR islemcide calisacak.
echo.
git log -1 --format="Guncel surum: %%h  %%s  (%%cr)"
pause
