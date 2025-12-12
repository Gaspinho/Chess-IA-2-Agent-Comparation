# Evaluación Visual Minimax vs MCTS

Este módulo permite ejecutar partidas de ajedrez entre los agentes Minimax y MCTS con visualización gráfica en tiempo real.

## 🚀 Inicio Rápido

### Instalación de dependencias
```bash
pip install -r requerimientos.txt
```

### Ejecutar evaluación visual
```bash
python ejecutar_minimax_vs_mcts.py
```

## 🎮 Formas de Ejecución

### 1. Script Simple (Recomendado)
```bash
python ejecutar_minimax_vs_mcts.py
```
- Interfaz interactiva para seleccionar configuración
- Configuraciones predefinidas optimizadas
- Instrucciones claras

### 2. Script Avanzado
```bash
# Configuración normal (predeterminada)
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_normal

# Configuración rápida
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_rapido

# Configuración fuerte
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_fuerte

# MCTS vs Minimax
python evaluar_minimax_mcts_visual.py --config mcts_vs_minimax_normal

# Configuración personalizada
python evaluar_minimax_mcts_visual.py --blancas minimax --negras mcts --profundidad 4 --simulaciones 1500 --velocidad 800
```

## ⚙️ Configuraciones Disponibles

### Rápida
- **Minimax**: Profundidad 2, 3s por jugada
- **MCTS**: 500 simulaciones, 3s por jugada
- **Velocidad**: 800ms entre jugadas

### Normal (Recomendada)
- **Minimax**: Profundidad 3, 5s por jugada
- **MCTS**: 1000 simulaciones, 5s por jugada
- **Velocidad**: 1000ms entre jugadas

### Fuerte
- **Minimax**: Profundidad 4, 8s por jugada
- **MCTS**: 2000 simulaciones, 8s por jugada
- **Velocidad**: 1200ms entre jugadas

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

### Visualización
- **Biblioteca**: pygame
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

### Partida Rápida para Demostración
```bash
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_rapido --velocidad 500
```

### Partida Competitiva
```bash
python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_fuerte
```

### Configuración Personalizada
```bash
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

### Rendimiento lento
- Usar configuración "rápida"
- Reducir profundidad de Minimax
- Reducir simulaciones de MCTS
- Aumentar velocidad de visualización

## 📈 Análisis Comparativo

### Ventajas de Minimax
- **Determinístico**: Misma entrada, misma salida
- **Profundidad garantizada**: Explora hasta nivel especificado
- **Evaluación heurística**: Usa conocimiento del dominio

### Ventajas de MCTS
- **Exploración balanceada**: UCB1 equilibra explotación y exploración
- **Escalabilidad**: Mejora con más simulaciones
- **Menos conocimiento específico**: Aprende patrones mediante simulación

### Observaciones Típicas
- **Aperturas**: Minimax suele ser más consistente
- **Medio juego**: MCTS puede encontrar tácticas creativas
- **Finales**: Minimax tiene ventaja con evaluación heurística
- **Tiempo de pensamiento**: MCTS es más predecible en duración

## 🚀 Próximas Mejoras

- [ ] Grabación de partidas en formato PGN
- [ ] Análisis post-partida con motor
- [ ] Múltiples partidas automáticas
- [ ] Gráficos de estadísticas
- [ ] Configuración de evaluación heurística
- [ ] Evaluación de AlphaZero vs Minimax/MCTS
- [ ] Soporte para agente PPO
- [ ] Interfaz gráfica para AlphaZero