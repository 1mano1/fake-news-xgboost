@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Instalando el entorno del proyecto
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro Python.
    echo Instala Python 3.10 o superior desde https://www.python.org/downloads/
    echo y marca la casilla "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creando el entorno virtual .venv ...
    python -m venv .venv
)

echo Instalando librerias (puede tardar un par de minutos)...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de librerias. Revisa tu conexion a internet.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Listo. Para trabajar abre una terminal en esta carpeta y ejecuta:
echo     .venv\Scripts\activate
echo ============================================
pause
