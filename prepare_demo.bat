@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Перед очисткой закройте окно run.bat (сервер ДОЗОР должен быть остановлен).
.venv\Scripts\python.exe tools\seed_demo.py --clear-live
echo Журнал очищен от тестовых тревог. Демо-история для аналитики осталась (камеры с пометкой «демо»).
echo Теперь запустите run.bat и обновите страницу в браузере.
pause
