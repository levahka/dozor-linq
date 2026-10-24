@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem --- ищем Python 3.11-3.13 (на 3.14+ ещё нет нужных библиотек) ---
set "PY="
py -3.13 -c "import sys" >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)" >nul 2>&1 && set "PY=python"
if not defined PY goto :nopython

rem --- окружение: пересоздаём, если оно сломано или не той версии ---
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import cv2, onnxruntime, fastapi, uvicorn, yaml, scipy" >nul 2>&1 || rmdir /s /q .venv
)
if not exist .venv (
  echo Создаю окружение и ставлю зависимости, 2-5 минут...
  %PY% -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install -U pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
)
echo.
echo Запускаю ДОЗОР. Браузер откроется сам, когда сервер будет готов (10-20 секунд).
set "CFG=config\config.yaml"
if exist data\my\demo\cam_fall.mp4 set "CFG=config\config_team.yaml"
echo Конфигурация: %CFG%
.venv\Scripts\python.exe -m sv.server --config %CFG% --port 8000 --open
pause
goto :eof

:nopython
echo Не найден Python 3.11-3.13. Установите Python 3.13 с python.org и запустите снова.
pause
goto :eof

:fail
echo Не удалось установить зависимости, см. сообщения выше.
pause
