# Guía de Instalación y Verificación - AlphaZero

## 📦 Instalación de Dependencias

### Paso 1: Instalar todas las dependencias
```bash
pip install -r requerimientos.txt
```

### Paso 2: Verificar instalación
```bash
python -c "import torch; import chess; import numpy; print('✓ Dependencias instaladas correctamente')"
```

### Instalación manual (si es necesario)
```bash
pip install torch torchvision
pip install chess
pip install numpy pandas matplotlib seaborn
pip install pygame
pip install pettingzoo[classic]
pip install stable-baselines3
pip install gymnasium
pip install tqdm
```

## ✅ Verificación de la Implementación

### 1. Verificar estructura de archivos
```bash
# En Windows (cmd)
dir agentes\agente_alphazero.py
dir entrenamiento\entrenar_alphazero.py
dir ejemplo_alphazero.py

# En Windows (PowerShell) o Linux/Mac
ls agentes/agente_alphazero.py
ls entrenamiento/entrenar_alphazero.py
ls ejemplo_alphazero.py
```

### 2. Verificar sintaxis de Python
```bash
python -m py_compile agentes/agente_alphazero.py
python -m py_compile entrenamiento/entrenar_alphazero.py
python -m py_compile principal.py
```

### 3. Probar ejemplos básicos
```bash
python ejemplo_alphazero.py
```

### 4. Verificar ayuda del CLI
```bash
python principal.py --help
```

Deberías ver `alphazero` como opción en `--entrenar`:
```
--entrenar {alphazero,ppo}
```

## 🚀 Primeros Pasos

### 1. Ejecutar Minimax vs MCTS (sin AlphaZero)
```bash
# Verificar que el proyecto base funciona
python ejecutar_minimax_vs_mcts.py
```

### 2. Probar ejemplos de AlphaZero
```bash
python ejemplo_alphazero.py
```

### 3. Entrenamiento de prueba (5 iteraciones)
```bash
python principal.py --entrenar alphazero --num-iteraciones 5 --num-episodios 10
```

**Nota**: Este entrenamiento de prueba tomará varios minutos. Es solo para verificar que todo funciona.

### 4. Entrenamiento real (recomendado)
```bash
# Para CPU (puede tomar horas)
python principal.py --entrenar alphazero --num-iteraciones 100

# Para GPU (más rápido)
python principal.py --entrenar alphazero --num-iteraciones 200 --num-simulaciones 1000
```

## 🔍 Solución de Problemas Comunes

### Error: "Import torch could not be resolved"

**Problema**: PyTorch no está instalado o no se encuentra.

**Solución**:
```bash
# Para CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Para GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Error: "Import chess could not be resolved"

**Problema**: python-chess no está instalado.

**Solución**:
```bash
pip install chess
```

### Error: "No module named 'entrenamiento'"

**Problema**: El directorio `entrenamiento` no existe o no tiene `__init__.py`.

**Solución**:
```bash
# Verificar que existe
dir entrenamiento\__init__.py   # Windows CMD
ls entrenamiento/__init__.py    # PowerShell/Linux/Mac

# Si no existe, crear
echo. > entrenamiento\__init__.py   # Windows CMD
touch entrenamiento/__init__.py     # PowerShell/Linux/Mac
```

### Error: "RuntimeError: CUDA out of memory"

**Problema**: GPU sin memoria suficiente.

**Solución**:
```bash
# Reducir batch size
python principal.py --entrenar alphazero --batch-size 32

# O forzar uso de CPU
# En el código, el agente automáticamente usa CPU si CUDA no está disponible
```

### Entrenamiento muy lento

**Problema**: Sin GPU o configuración demasiado intensiva.

**Soluciones**:
1. Reducir simulaciones:
```bash
python principal.py --entrenar alphazero --num-simulaciones 400
```

2. Reducir episodios por iteración:
```bash
python principal.py --entrenar alphazero --num-episodios 50
```

3. Entrenamiento mínimo para pruebas:
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 10 \
    --num-episodios 10 \
    --num-simulaciones 200
```

## 📊 Monitoreo del Entrenamiento

### Ver progreso en tiempo real
```bash
python principal.py --entrenar alphazero --verbose
```

### Verificar modelos guardados
```bash
# Windows CMD
dir modelos\*.pth

# PowerShell/Linux/Mac
ls modelos/*.pth
```

Los modelos se guardan como:
- `alphazero_iter_10.pth` (cada 10 iteraciones por defecto)
- `alphazero_final.pth` (al terminar)

### Cargar y probar modelo entrenado
```python
from agentes.agente_alphazero import AgenteAlphaZero

agente = AgenteAlphaZero(
    modelo_path='modelos/alphazero_final.pth',
    num_simulaciones=800
)

print("Modelo cargado:", agente.nombre)
print("Dispositivo:", agente.dispositivo)
```

## 🎮 Próximos Pasos

### 1. Entrenar modelo básico
```bash
python principal.py --entrenar alphazero --num-iteraciones 50
```

### 2. Evaluar contra otros agentes (cuando esté implementado)
```bash
python principal.py --evaluar
```

### 3. Experimentar con parámetros
```bash
# Más exploración
python principal.py --entrenar alphazero --temperatura 1.5

# Más simulaciones MCTS
python principal.py --entrenar alphazero --num-simulaciones 1600

# Aprendizaje más lento pero estable
python principal.py --entrenar alphazero --learning-rate 0.0005
```

## 📚 Recursos Adicionales

### Documentación
- `ALPHAZERO_INFO.md`: Documentación técnica completa
- `CAMBIOS_DQN_A_ALPHAZERO.md`: Resumen de cambios realizados
- `README.md`: Guía general del proyecto

### Archivos de ejemplo
- `ejemplo_alphazero.py`: Ejemplos de uso práctico
- `agentes/agente_alphazero.py`: Implementación del agente
- `entrenamiento/entrenar_alphazero.py`: Sistema de entrenamiento

## ⚙️ Configuración Recomendada

### Para desarrollo/pruebas (rápido)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 10 \
    --num-episodios 20 \
    --num-simulaciones 400 \
    --batch-size 32
```

**Tiempo estimado**: 30-60 minutos en CPU moderna

### Para entrenamiento serio (recomendado)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 100 \
    --num-episodios 100 \
    --num-simulaciones 800 \
    --batch-size 64
```

**Tiempo estimado**: Varias horas en CPU, ~1-2 horas en GPU

### Para investigación (intensivo)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 500 \
    --num-episodios 200 \
    --num-simulaciones 1600 \
    --batch-size 128 \
    --learning-rate 0.0005
```

**Tiempo estimado**: Días en CPU, ~8-12 horas en GPU potente

## ✨ Verificación Final

Ejecuta este comando para verificar que todo está listo:

```bash
python -c "
import sys
import os

print('=' * 60)
print('VERIFICACIÓN DE INSTALACIÓN - ALPHAZERO')
print('=' * 60)

# Verificar Python
print(f'✓ Python {sys.version.split()[0]}')

# Verificar dependencias
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  - CUDA disponible: {torch.cuda.is_available()}')
except ImportError:
    print('✗ PyTorch no instalado')

try:
    import chess
    print('✓ python-chess instalado')
except ImportError:
    print('✗ python-chess no instalado')

try:
    import numpy
    print('✓ NumPy instalado')
except ImportError:
    print('✗ NumPy no instalado')

# Verificar archivos
archivos = [
    'agentes/agente_alphazero.py',
    'entrenamiento/entrenar_alphazero.py',
    'entrenamiento/__init__.py',
    'ejemplo_alphazero.py',
    'principal.py'
]

print()
print('Archivos del proyecto:')
for archivo in archivos:
    if os.path.exists(archivo):
        print(f'✓ {archivo}')
    else:
        print(f'✗ {archivo} - NO ENCONTRADO')

print()
print('=' * 60)
print('Listo para empezar!')
print('Comando de prueba: python ejemplo_alphazero.py')
print('=' * 60)
"
```

Si ves checkmarks (✓) en todo, ¡estás listo para empezar! 🎉
