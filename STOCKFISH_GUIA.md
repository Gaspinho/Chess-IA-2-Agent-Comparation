# Guía de Instalación y Uso de Stockfish

## 🎯 Introducción

Stockfish es uno de los motores de ajedrez más potentes del mundo. Esta implementación permite que Stockfish compita contra los otros agentes del proyecto (Minimax, MCTS, AlphaZero).

## 📦 Instalación de Stockfish

### Windows

1. **Descarga Stockfish**:
   - Ve a [stockfishchess.org/download](https://stockfishchess.org/download/)
   - Descarga la versión para Windows
   - Extrae el archivo `stockfish.exe`

2. **Opción 1 - Agregar al PATH**:
   ```cmd
   # Copia stockfish.exe a C:\Windows\System32\
   # O añade la carpeta de Stockfish a las variables de entorno PATH
   ```

3. **Opción 2 - Ruta específica**:
   ```bash
   # Guarda stockfish.exe en una ubicación y usa --ruta-stockfish
   python ejecutar_stockfish.py --ruta-stockfish "C:\ruta\a\stockfish.exe"
   ```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install stockfish

# Fedora
sudo dnf install stockfish

# Arch Linux
sudo pacman -S stockfish
```

### macOS

```bash
# Usando Homebrew
brew install stockfish

# Usando MacPorts
sudo port install stockfish
```

## 🚀 Uso Básico

### Stockfish vs Minimax

```bash
# Configuración básica
python ejecutar_stockfish.py --blancas stockfish --negras minimax

# Stockfish débil (nivel 5) vs Minimax
python ejecutar_stockfish.py --blancas stockfish --negras minimax --nivel-stockfish 5

# Stockfish fuerte (nivel 20) vs Minimax profundo
python ejecutar_stockfish.py --blancas stockfish --negras minimax --nivel-stockfish 20 --profundidad-minimax 4
```

### Stockfish vs MCTS

```bash
# Configuración equilibrada
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 10 --simulaciones-mcts 1000

# Stockfish rápido vs MCTS con muchas simulaciones
python ejecutar_stockfish.py --blancas stockfish --negras mcts --tiempo-stockfish 0.5 --simulaciones-mcts 2000
```

### Minimax vs Stockfish

```bash
# Minimax con blancas
python ejecutar_stockfish.py --blancas minimax --negras stockfish --nivel-stockfish 15
```

### MCTS vs Stockfish

```bash
# MCTS con blancas
python ejecutar_stockfish.py --blancas mcts --negras stockfish --simulaciones-mcts 1500
```

## ⚙️ Parámetros de Configuración

### Parámetros de Stockfish

| Parámetro | Valores | Descripción |
|-----------|---------|-------------|
| `--nivel-stockfish` | 0-20 | Nivel de juego (0=débil, 20=máximo) |
| `--tiempo-stockfish` | segundos | Tiempo límite por jugada |
| `--ruta-stockfish` | ruta | Ruta al ejecutable de Stockfish |

### Parámetros de Minimax

| Parámetro | Valores | Descripción |
|-----------|---------|-------------|
| `--profundidad-minimax` | 1-6 | Profundidad de búsqueda |
| `--tiempo-minimax` | segundos | Tiempo límite por jugada |

### Parámetros de MCTS

| Parámetro | Valores | Descripción |
|-----------|---------|-------------|
| `--simulaciones-mcts` | número | Número de simulaciones |
| `--tiempo-mcts` | segundos | Tiempo límite por jugada |

### Parámetros Generales

| Parámetro | Valores | Descripción |
|-----------|---------|-------------|
| `--max-jugadas` | número | Máximo de jugadas (default: 200) |

## 📊 Niveles de Stockfish

Los niveles de Stockfish (0-20) determinan la fuerza del motor:

| Nivel | Aproximado ELO | Descripción |
|-------|----------------|-------------|
| 0-3 | <1000 | Principiante |
| 4-7 | 1000-1400 | Intermedio bajo |
| 8-11 | 1400-1800 | Intermedio |
| 12-15 | 1800-2200 | Avanzado |
| 16-18 | 2200-2600 | Maestro |
| 19-20 | 2600+ | Super Gran Maestro |

## 🎮 Ejemplos de Uso

### 1. Prueba Rápida

```bash
# Stockfish débil vs Minimax (5 minutos aprox)
python ejecutar_stockfish.py --nivel-stockfish 5 --tiempo-stockfish 0.3 --profundidad-minimax 2
```

### 2. Partida Equilibrada

```bash
# Stockfish medio vs MCTS (15-20 minutos)
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 10 --simulaciones-mcts 1000
```

### 3. Análisis de Calidad

```bash
# Stockfish fuerte vs Minimax fuerte (30+ minutos)
python ejecutar_stockfish.py --nivel-stockfish 18 --tiempo-stockfish 2.0 --profundidad-minimax 5 --tiempo-minimax 10.0
```

### 4. Benchmark

```bash
# Stockfish máximo vs todos los agentes
python ejecutar_stockfish.py --blancas stockfish --negras minimax --nivel-stockfish 20 --tiempo-stockfish 3.0
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 20 --tiempo-stockfish 3.0
```

## 🔍 Verificación de Instalación

Ejecuta este comando para verificar que Stockfish está correctamente instalado:

```bash
# Windows
where stockfish

# Linux/macOS
which stockfish

# Probar directamente
stockfish
```

Si ves la salida de Stockfish (incluyendo "Stockfish" y versión), está correctamente instalado.

## 🛠️ Uso Programático

También puedes usar el agente Stockfish directamente en tu código:

```python
from agentes.agente_stockfish import AgenteStockfish

# Crear agente
agente = AgenteStockfish(
    nivel_habilidad=15,      # Nivel 0-20
    tiempo_limite=1.0,       # 1 segundo por jugada
    threads=2,               # Usar 2 hilos
    hash_size=256            # 256 MB de tabla hash
)

# Usar en una partida
# ... (ver ejecutar_stockfish.py para ejemplo completo)

# Cerrar cuando termines
agente.cerrar()
```

## 📈 Análisis de Posiciones

El agente también puede evaluar posiciones sin jugar:

```python
import chess
from agentes.agente_stockfish import AgenteStockfish

agente = AgenteStockfish()
tablero = chess.Board()

# Evaluar posición inicial
evaluacion = agente.obtener_evaluacion(tablero, tiempo=0.5)
print(f"Evaluación: {evaluacion} centipeones")

agente.cerrar()
```

## ❓ Solución de Problemas

### Error: "No se pudo encontrar el ejecutable de Stockfish"

**Solución**:
1. Verifica que Stockfish esté instalado
2. Usa `--ruta-stockfish` con la ruta completa
3. En Windows, asegúrate de que el archivo se llame `stockfish.exe`

### Stockfish muy lento

**Solución**:
- Reduce `--nivel-stockfish`
- Reduce `--tiempo-stockfish`
- Usa `profundidad` en lugar de `tiempo_limite`:
  ```python
  agente = AgenteStockfish(profundidad=10, tiempo_limite=None)
  ```

### Error de UCI

**Solución**:
- Asegúrate de tener la versión correcta de Stockfish (12+)
- Verifica que el ejecutable no esté corrupto
- Descarga nuevamente desde el sitio oficial

## 🔗 Referencias

- [Sitio oficial de Stockfish](https://stockfishchess.org/)
- [Documentación python-chess](https://python-chess.readthedocs.io/)
- [UCI Protocol](https://www.shredderchess.com/download/div/uci.zip)
