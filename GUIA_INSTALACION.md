# 🧠 Guía de Instalación - Sistema Experto Difuso

## 📋 Requisitos Previos

### 1. **Python (OBLIGATORIO)**
- **Descargar**: Ve a [python.org](https://www.python.org/downloads/)
- **Versión**: Python 3.8 o superior
- **IMPORTANTE**: Durante la instalación, marca la casilla **"Add Python to PATH"**

### 2. **Verificar Instalación**
- Abre la terminal (Windows: `cmd` o `PowerShell`)
- Escribe: `python --version`
- Debe mostrar algo como: `Python 3.11.x` o superior

## 🚀 Instalación Paso a Paso

### **PASO 1: Descargar el Archivo**
1. Descarga el archivo `fuzzy_system_complete.py`
2. Guárdalo en una carpeta (ej: `C:\SistemaDifuso\`)

### **PASO 2: Instalar Módulos Necesarios**
Abre la terminal y ejecuta estos comandos **UNO POR UNO**:

```bash
pip install panel
pip install pydantic
pip install numpy
pip install scipy
```

**Si tienes problemas, prueba con:**
```bash
python -m pip install panel pydantic numpy scipy
```

### **PASO 3: Ejecutar el Sistema**
1. Abre la terminal
2. Navega a la carpeta donde guardaste el archivo:
   ```bash
   cd C:\SistemaDifuso
   ```
3. Ejecuta el sistema:
   ```bash
   python fuzzy_system_complete.py
   ```

### **PASO 4: Abrir en el Navegador**
- El sistema se abrirá automáticamente en: `http://localhost:5011`
- Si no se abre automáticamente, copia y pega esa dirección en tu navegador

## 🔧 Solución de Problemas Comunes

### **Error: "python no se reconoce como comando"**
- **Solución**: Python no está en el PATH
- **Reinstalar Python** marcando "Add Python to PATH"

### **Error: "pip no se reconoce como comando"**
- **Solución**: Usar `python -m pip` en lugar de `pip`

### **Error: "Puerto en uso"**
- **Solución**: Cerrar otras ventanas del navegador con el sistema
- **O cambiar el puerto** en el archivo (línea final)

### **Error: "Módulo no encontrado"**
- **Solución**: Instalar el módulo faltante:
  ```bash
  pip install nombre_del_modulo
  ```

## 📱 Uso del Sistema

### **Panel Izquierdo - Control de Entrada**
- **Sliders**: Mueve para cambiar valores de entrada
- **Botón "Ejecutar Inferencia"**: Calcula resultados
- **Resultados**: Muestra valores numéricos y interpretación

### **Panel Derecho - Editores**
- **Pestaña "Variables"**: Crear/editar variables del sistema
- **Pestaña "Reglas"**: Crear/editar reglas difusas
- **Pestaña "Gráficas"**: Ver funciones de pertenencia
- **Pestaña "Ayuda"**: Guía de parámetros

## 🎯 Ejemplos de Uso

### **Crear una Variable**
1. Ve a "Variables" → "Agregar Variable"
2. Nombre: `Temperatura`
3. Tipo: `input`
4. Universo: `0` a `100`
5. Términos: `Bajo`, `Medio`, `Alto`

### **Crear una Regla**
1. Ve a "Reglas" → "Agregar Regla"
2. ID: `R1`
3. Condición: `Temperatura is Alto`
4. Conclusión: `Riesgo is Alto`

## 📞 Soporte

### **Si algo no funciona:**
1. **Verifica Python**: `python --version`
2. **Verifica módulos**: `pip list`
3. **Revisa errores**: Copia el mensaje de error completo
4. **Reinicia**: Cierra todo y vuelve a intentar

### **Comandos de Verificación:**
```bash
# Verificar Python
python --version

# Verificar módulos instalados
pip list

# Verificar archivo
dir fuzzy_system_complete.py
```

## 🎉 ¡Listo!

Una vez que veas el mensaje:
```
Launching server at http://localhost:5011
```

**¡El sistema está funcionando correctamente!** 🚀

---

## 📝 Notas Importantes

- **No cierres la terminal** mientras uses el sistema
- **El sistema funciona offline** una vez instalado
- **Para detener**: Presiona `Ctrl+C` en la terminal
- **Para reiniciar**: Ejecuta `python fuzzy_system_complete.py` nuevamente

**¡Disfruta explorando el sistema experto difuso!** 🧠✨

