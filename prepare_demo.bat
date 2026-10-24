@echo off
chcp 65001 >nul
cd /d "%~dp0"
.venv\Scripts\python.exe tools\seed_demo.py --clear-live
echo Журнал очищен от тестовых событий, демо-история осталась.
pause
