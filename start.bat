@echo off
rem Hyllora anonymizer for testing on Windows with an NVIDIA card. Double-click to start.
rem The first time it installs uv, Python 3.12, torch for the graphics card and the three models: a few GB.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo Installerar uv, som haemtar Python ...
  powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  if errorlevel 1 goto fail
)

if exist .venv\ready goto run
echo.
echo Foersta gaangen: Python, torch foer grafikkortet och modellerna. Det tar en stund.
echo.
if exist .venv rmdir /s /q .venv
uv venv --python 3.12 .venv
if errorlevel 1 goto fail
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
uv pip install --python .venv\Scripts\python.exe -r packaging\requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe get_models.py
if errorlevel 1 goto fail
echo ok> .venv\ready

:run
.venv\Scripts\python.exe -c "import torch; print('Grafikkort:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'INGET, koer paa processorn. Uppdatera NVIDIA-drivrutinen.')"
echo.
echo Anonymiseraren koer paa http://127.0.0.1:8430  -  staeng det haer foenstret foer att stoppa den.
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8430"
.venv\Scripts\python.exe -m uvicorn anonymizer.server:app --host 127.0.0.1 --port 8430
goto end

:fail
echo.
echo Naagot gick fel. Ta en skaermbild av foenstret och skicka den till Sameh.
pause
:end
