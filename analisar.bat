@echo off
rem Análise tática de futebol: dois cliques para escolher um vídeo,
rem ou arraste um vídeo para cima deste arquivo.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente nao encontrado. Instale primeiro, conforme o README:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" src\analisar.py %*
echo.
pause
