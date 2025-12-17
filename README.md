# Comparación de Agentes de Ajedrez con IA

Este proyecto permite ejecutar y comparar diferentes agentes de Inteligencia Artificial jugando ajedrez, incluyendo Minimax, MCTS, AlphaZero y **Stockfish**.

## 🚀 Inicio Rápido

### Instalación de dependencias
```bash
pip install -r requerimientos.txt
```

### Instalar Stockfish (opcional pero recomendado)
- **Windows**: Descarga desde [stockfishchess.org/download](https://stockfishchess.org/download/)
- **Linux**: `sudo apt-get install stockfish`
- **macOS**: `brew install stockfish`

Ver [STOCKFISH_GUIA.md](STOCKFISH_GUIA.md) para instrucciones detalladas.

### Ejecutar evaluaciones
```bash
# Minimax vs MCTS (original)
python ejecutar_minimax_vs_mcts.py

# Stockfish vs Minimax
python ejecutar_stockfish.py --blancas stockfish --negras minimax

# Stockfish vs MCTS
python ejecutar_stockfish.py --blancas stockfish --negras mcts
```

## 🎮 Formas de Ejecución

### 1. Minimax vs MCTS (Original)
```bash
# Script simple
python ejecutar_minimax_vs_mcts.py

# Script avanzado con visualización
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_normal
```

### 2. **Stockfish vs Otros Agentes** ⭐ NUEVO
```bash
# Stockfish vs Minimax
python ejecutar_stockfish.py --blancas stockfish --negras minimax --nivel-stockfish 10

# Stockfish vs MCTS
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 15

# Minimax vs Stockfish
python ejecutar_stockfish.py --blancas minimax --negras stockfish

# Personalizar configuración
python ejecutar_stockfish.py --blancas stockfish --negras minimax \
    --nivel-stockfish 15 \
    --tiempo-stockfish 1.0 \
    --profundidad-minimax 4 \
    --tiempo-minimax 5.0
```

### 3. Configuraciones Predefinidas (Minimax/MCTS)
```bash
# Configuración rápida
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_rapido

# Configuración fuerte
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_fuerte

# Configuración personalizada
python evaluar_minimax_mcts_visual.py --blancas minimax --negras mcts --profundidad 4 --simulaciones 1500 --velocidad 800
```

## ⚙️ Configuraciones Disponibles

### Stockfish
- **Nivel 0-5**: Principiante (~800-1200 ELO)
- **Nivel 10**: Intermedio (~1800 ELO)
- **Nivel 15**: Avanzado (~2200 ELO)
- **Nivel 20**: Gran Maestro (~3000+ ELO)
- **Tiempo**: 0.1s - 5s por jugada

### Minimax
- **Rápida**: Profundidad 2, 3s por jugada
- **Normal**: Profundidad 3, 5s por jugada
- **Fuerte**: Profundidad 4-5, 8s por jugada

### MCTS
- **Rápida**: 500 simulaciones, 3s por jugada
- **Normal**: 1000 simulaciones, 5s por jugada
- **Fuerte**: 2000 simulaciones, 8s por jugada

## 🎯 Controles Durante la Partida

| Tecla | Acción |
|-------|--------|
| `ESC` | Salir del juego |
| `SPACE` | Pausar/Reanudar |
| `R` | Reiniciar (futuro) |

## 📊 Información Mostrada

### Panel de Información
- **Agentes**: Nombres de los agentes jugando
- **Turno actual**: Quién debe mover
- **Estado del juego**: Jaque, jaque mate, empate, etc.
- **Estadísticas en tiempo real**:
  - Número de jugadas
  - Tiempo por agente
  - Última jugada realizada
  - Tiempo de la última jugada

### Tablero Visual
- **Casillas destacadas**: Última jugada realizada
- **Jaque destacado**: Rey en jaque resaltado en rojo
- **Piezas**: Símbolos Unicode para las piezas

## 🏆 Análisis de Resultados

Al finalizar cada partida, se muestra:
- **Ganador** o tipo de empate
- **Número total de jugadas**
- **Tiempo total de la partida**
- **Tiempo usado por cada agente**
- **Análisis básico del resultado**

## 🛠️ Características Técnicas

### Algoritmo Minimax
- Búsqueda con poda alfa-beta
- Evaluación heurística de posiciones
- Control de profundidad y tiempo
- Estadísticas de nodos evaluados

### Algoritmo MCTS
- Simulaciones Monte Carlo
- Selección UCB1
- Expansión y retropropagación
- Control de número de simulaciones y tiempo

### **Stockfish** ⭐
- Motor UCI de clase mundial
- Configuración de nivel (0-20)
- Evaluación en centipeones
- Análisis de posiciones
- Multi-threading
- Tabla hash configurable

### Visualización
- **Biblioteca**: pygame (opcional)
- **Renderizado**: 60 FPS
- **Resolución**: Adaptable al tamaño de casilla
- **Interfaz**: Tiempo real con información detallada

## 🤖 AlphaZero (Implementado)

AlphaZero combina MCTS con redes neuronales profundas para aprendizaje mediante auto-juego:

### Características Principales
- **Auto-juego**: Genera datos jugando contra sí mismo
- **Red neuronal**: Evalúa posiciones y guía MCTS
- **Arquitectura**: Torre residual con cabezas de política y valor
- **Entrenamiento iterativo**: Mejora continua mediante datos de auto-juego

### Entrenar AlphaZero
```bash
# Entrenamiento básico (100 iteraciones)
python principal.py --entrenar alphazero

# Entrenamiento completo personalizado
python principal.py --entrenar alphazero \
    --num-iteraciones 200 \
    --num-simulaciones 800 \
    --num-episodios 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --temperatura 1.0 \
    --c-puct 1.5
```

### Parámetros de AlphaZero
- `--num-iteraciones`: Ciclos de entrenamiento (default: 100)
- `--num-simulaciones`: Simulaciones MCTS por jugada (default: 800)
- `--num-episodios`: Partidas por iteración (default: 100)
- `--temperatura`: Exploración (1.0 = alta, 0.0 = baja)
- `--c-puct`: Constante exploración PUCT (default: 1.5)
- `--batch-size`: Tamaño de batch para entrenamiento (default: 64)
- `--learning-rate`: Tasa de aprendizaje (default: 0.001)

### Ver AlphaZero Jugar en Pantalla 🎮

Puedes ver a AlphaZero jugando contra otros agentes en tiempo real:

```bash
# Script interactivo (recomendado)
python ver_alphazero_jugar.py

# Opciones disponibles:
# 1. AlphaZero vs Minimax
# 2. AlphaZero vs MCTS  
# 3. Minimax vs AlphaZero
# 4. MCTS vs AlphaZero
# 5. AlphaZero vs AlphaZero (auto-juego)
```

**Controles durante la partida:**
- `ESC` - Salir
- `SPACE` - Pausar/Reanudar

**Nota:** AlphaZero jugará con red neuronal aleatoria hasta entrenar un modelo. Para mejores resultados, entrena primero con `python principal.py --entrenar alphazero`

## 📝 Ejemplos de Uso

### Partidas con Stockfish

```bash
# Prueba rápida - Stockfish débil vs Minimax
python ejecutar_stockfish.py --nivel-stockfish 5 --profundidad-minimax 2

# Partida equilibrada - Stockfish medio vs MCTS
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 10 --simulaciones-mcts 1000

# Desafío difícil - Stockfish fuerte vs Minimax
python ejecutar_stockfish.py --nivel-stockfish 18 --profundidad-minimax 5
```

### Partidas Minimax vs MCTS

```bash
# Partida rápida para demostración
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_rapido --velocidad 500

# Partida competitiva
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_fuerte

# Configuración personalizada
python evaluar_minimax_mcts_visual.py --blancas mcts --negras minimax --profundidad 3 --simulaciones 1200 --tiempo-limite 6.0 --velocidad 800
```

## 🔧 Solución de Problemas

### Pygame no instalado
```bash
pip install pygame
```

### Chess library no encontrada
```bash
pip install chess
```

### Error de dependencias
```bash
pip install -r requerimientos.txt
```

### Stockfish no encontrado
Ver [STOCKFISH_GUIA.md](STOCKFISH_GUIA.md) para instrucciones de instalación detalladas.

### Rendimiento lento
- Usar configuración "rápida"
- Reducir nivel de Stockfish (0-10)
- Reducir profundidad de Minimax
- Reducir simulaciones de MCTS
- Aumentar velocidad de visualización

## 📈 Análisis Comparativo

### Stockfish ⭐
- **Máxima fuerza**: Motor de ~3000+ ELO en nivel máximo
- **Evaluación precisa**: Centipeones exactos
- **Highly optimized**: Décadas de optimización
- **Configurable**: Adapta nivel de 800 a 3000+ ELO

### Minimax
- **Determinístico**: Misma entrada, misma salida
- **Profundidad garantizada**: Explora hasta nivel especificado
- **Evaluación heurística**: Usa conocimiento del dominio
- **Transparente**: Fácil de entender y depurar

### MCTS
- **Exploración balanceada**: UCB1 equilibra explotación y exploración
- **Escalabilidad**: Mejora con más simulaciones
- **Menos conocimiento específico**: Aprende patrones mediante simulación
- **Robusto**: Funciona sin evaluación heurística

### AlphaZero
- **Aprendizaje autónomo**: Mejora jugando contra sí mismo
- **Evaluación neural**: Red profunda evalúa posiciones
- **MCTS guiado**: Usa red para guiar búsqueda
- **Estado del arte**: Similar al enfoque de Google DeepMind

### Comparación de Fuerza Aproximada
1. **Stockfish (nivel 20)**: ~3000+ ELO
2. **AlphaZero (entrenado)**: ~2500-3000 ELO
3. **Stockfish (nivel 15)**: ~2200 ELO
4. **MCTS (2000 sim)**: ~1800 ELO
5. **Minimax (prof 5)**: ~1500 ELO
6. **Stockfish (nivel 5)**: ~1200 ELO

## � Archivos Principales

### Scripts de Ejecución
- `ejecutar_minimax_vs_mcts.py` - Partidas Minimax vs MCTS (original)
- `ejecutar_stockfish.py` - **Partidas con Stockfish** ⭐
- `evaluar_minimax_mcts_visual.py` - Visualización con pygame
- `ejemplo_stockfish.py` - Ejemplos de uso programático
- `verificar_stockfish.py` - Verificar instalación de Stockfish

### Agentes
- `agentes/agente_minimax.py` - Minimax con poda alfa-beta
- `agentes/agente_mcts.py` - Monte Carlo Tree Search
- `agentes/agente_stockfish.py` - **Motor Stockfish UCI** ⭐
- `agentes/agente_alphazero.py` - AlphaZero (MCTS + NN)
- `agentes/agente_ppo.py` - Proximal Policy Optimization

### Documentación
- `README.md` - Este archivo
- `STOCKFISH_GUIA.md` - **Guía completa de Stockfish** ⭐
- `IMPLEMENTACION_STOCKFISH.md` - Detalles técnicos
- `ALPHAZERO_INFO.md` - Información de AlphaZero
- `INICIO_RAPIDO.md` - Guía de inicio rápido

## 🚀 Próximas Mejoras

- [x] **Integración de Stockfish** ✅
- [x] **Configuración de niveles de Stockfish** ✅
- [ ] Visualización con Stockfish (pygame)
- [ ] Torneo round-robin entre todos los agentes
- [ ] Grabación de partidas en formato PGN
- [ ] Análisis post-partida con Stockfish
- [ ] Múltiples partidas automáticas con estadísticas
- [ ] Gráficos de estadísticas comparativas
- [ ] AlphaZero entrenado vs Stockfish
- [ ] Interfaz web con Flask/Streamlit