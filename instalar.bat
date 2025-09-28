@echo off
echo ========================================
echo    INSTALADOR SISTEMA EXPERTO DIFUSO
echo ========================================
echo.

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo.
    echo Por favor:
    echo 1. Descarga Python desde https://www.python.org/downloads/
    echo 2. Durante la instalacion, marca "Add Python to PATH"
    echo 3. Reinicia esta ventana y ejecuta nuevamente
    echo.
    pause
    exit /b 1
)

echo Python encontrado!
echo.

echo Instalando modulos necesarios...
echo.

echo Instalando Panel...
pip install panel
if errorlevel 1 (
    echo ERROR al instalar Panel
    pause
    exit /b 1
)

echo Instalando Pydantic...
pip install pydantic
if errorlevel 1 (
    echo ERROR al instalar Pydantic
    pause
    exit /b 1
)

echo Instalando NumPy...
pip install numpy
if errorlevel 1 (
    echo ERROR al instalar NumPy
    pause
    exit /b 1
)

echo Instalando SciPy...
pip install scipy
if errorlevel 1 (
    echo ERROR al instalar SciPy
    pause
    exit /b 1
)

echo.
echo ========================================
echo    INSTALACION COMPLETADA EXITOSAMENTE
echo ========================================
echo.
echo Para ejecutar el sistema:
echo 1. Ejecuta: python fuzzy_system_complete.py
echo 2. Abre tu navegador en: http://localhost:5011
echo.
echo Presiona cualquier tecla para ejecutar el sistema ahora...
pause >nul

echo.
echo Ejecutando Sistema Experto Difuso...
echo.
python fuzzy_system_complete.py

