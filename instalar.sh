#!/bin/bash

echo "========================================"
echo "   INSTALADOR SISTEMA EXPERTO DIFUSO"
echo "========================================"
echo

echo "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 no está instalado"
    echo
    echo "Por favor instala Python3:"
    echo "- Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "- macOS: brew install python3"
    echo "- CentOS/RHEL: sudo yum install python3 python3-pip"
    echo
    exit 1
fi

echo "Python encontrado!"
echo

echo "Instalando módulos necesarios..."
echo

echo "Instalando Panel..."
pip3 install panel
if [ $? -ne 0 ]; then
    echo "ERROR al instalar Panel"
    exit 1
fi

echo "Instalando Pydantic..."
pip3 install pydantic
if [ $? -ne 0 ]; then
    echo "ERROR al instalar Pydantic"
    exit 1
fi

echo "Instalando NumPy..."
pip3 install numpy
if [ $? -ne 0 ]; then
    echo "ERROR al instalar NumPy"
    exit 1
fi

echo "Instalando SciPy..."
pip3 install scipy
if [ $? -ne 0 ]; then
    echo "ERROR al instalar SciPy"
    exit 1
fi

echo
echo "========================================"
echo "   INSTALACIÓN COMPLETADA EXITOSAMENTE"
echo "========================================"
echo
echo "Para ejecutar el sistema:"
echo "1. Ejecuta: python3 fuzzy_system_complete.py"
echo "2. Abre tu navegador en: http://localhost:5011"
echo
echo "¿Deseas ejecutar el sistema ahora? (y/n)"
read -r respuesta

if [[ $respuesta =~ ^[Yy]$ ]]; then
    echo
    echo "Ejecutando Sistema Experto Difuso..."
    echo
    python3 fuzzy_system_complete.py
fi

