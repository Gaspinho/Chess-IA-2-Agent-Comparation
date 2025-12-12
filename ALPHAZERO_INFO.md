# AlphaZero - Implementación para Ajedrez

## 🎯 Descripción General

Esta implementación de AlphaZero reemplaza el agente DQN del proyecto original. AlphaZero es significativamente más apropiado para ajedrez ya que combina:

- **Monte Carlo Tree Search (MCTS)**: Para exploración eficiente del espacio de jugadas
- **Redes Neuronales Profundas**: Para evaluación de posiciones y políticas de movimiento
- **Auto-juego**: Aprendizaje sin necesidad de datos externos o supervisión humana

## 🏗️ Arquitectura

### Red Neuronal

La red neuronal de AlphaZero usa una arquitectura de torre residual:

```
Entrada (8x8x119) 
    ↓
Conv2D + BatchNorm + ReLU
    ↓
Bloques Residuales (x10)
    ↓         ↓
Cabeza de    Cabeza de
Política     Valor
    ↓         ↓
Softmax      Tanh
```

**Entrada**: Representación del tablero en 119 planos (piezas, turno, castling, etc.)

**Salida doble**:
1. **Política**: Distribución de probabilidad sobre 4672 movimientos posibles
2. **Valor**: Evaluación escalar de la posición (-1 a +1)

### Componentes Principales

#### 1. `RedNeuralAlphaZero` (agente_alphazero.py)
- Torre de bloques residuales convolucionales
- Dos cabezas: política y valor
- Parámetros configurables (canales, bloques)

#### 2. `AgenteAlphaZero` (agente_alphazero.py)
- Combina MCTS con predicciones de la red
- Gestiona selección de jugadas
- Maneja estadísticas y modelo

#### 3. `EntrenadorAlphaZero` (entrenamiento/entrenar_alphazero.py)
- Ciclo de auto-juego
- Entrenamiento de la red con datos generados
- Guardado periódico de modelos

## 🔄 Proceso de Entrenamiento

### Ciclo Iterativo

```
1. AUTO-JUEGO (Self-Play)
   ├─ Jugar N partidas completas
   ├─ Ejecutar MCTS guiado por red neuronal
   ├─ Recolectar (estado, política_MCTS, resultado)
   └─ Agregar a buffer de experiencia

2. ENTRENAMIENTO
   ├─ Muestrear batches del buffer
   ├─ Calcular pérdidas:
   │  ├─ Pérdida de política (cross-entropy)
   │  └─ Pérdida de valor (MSE)
   ├─ Backpropagation
   └─ Actualizar pesos de la red

3. EVALUACIÓN (opcional)
   └─ Comparar con versión anterior

4. REPETIR
```

### Parámetros Clave

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `num_iteraciones` | 100 | Ciclos de entrenamiento |
| `num_episodios` | 100 | Partidas por iteración |
| `num_simulaciones` | 800 | Simulaciones MCTS por jugada |
| `batch_size` | 64 | Tamaño de batch para entrenamiento |
| `learning_rate` | 0.001 | Tasa de aprendizaje |
| `temperatura` | 1.0 | Control de exploración (↓ = explotación) |
| `c_puct` | 1.5 | Constante exploración PUCT |

## 🎮 Uso

### Entrenar desde Cero

```bash
# Entrenamiento básico
python principal.py --entrenar alphazero

# Entrenamiento personalizado
python principal.py --entrenar alphazero \
    --num-iteraciones 200 \
    --num-simulaciones 1000 \
    --batch-size 128 \
    --learning-rate 0.0005
```

### Usar Modelo Entrenado

```python
from agentes.agente_alphazero import AgenteAlphaZero

# Cargar modelo
agente = AgenteAlphaZero(
    num_simulaciones=800,
    modelo_path='modelos/alphazero_final.pth'
)

# Usar para jugar
accion = agente.seleccionar_accion(observacion, acciones_legales, info)
```

### Ejemplos Interactivos

```bash
# Ver ejemplos de uso
python ejemplo_alphazero.py
```

## 📊 Ventajas sobre DQN

| Aspecto | DQN | AlphaZero |
|---------|-----|-----------|
| **Búsqueda** | No planifica | MCTS + Red neuronal |
| **Datos** | Necesita experiencia variada | Auto-juego |
| **Exploración** | ε-greedy simple | MCTS balanceado |
| **Convergencia** | Más lenta | Más rápida con auto-juego |
| **Calidad** | Limitada para ajedrez | Estado del arte |

### ¿Por qué AlphaZero es mejor para ajedrez?

1. **Planificación Explícita**: MCTS busca activamente en el árbol de jugadas
2. **Sin necesidad de oponentes externos**: Se auto-mejora jugando contra sí mismo
3. **Evaluación robusta**: Red neuronal aprende patrones complejos
4. **Escalabilidad**: Mejora con más simulaciones y entrenamiento
5. **Generalización**: Aprende estrategias sin conocimiento específico del dominio

## 🔧 Implementación Técnica

### Representación del Tablero

El tablero se codifica en **119 planos de 8x8**:

- **0-11**: Posiciones de piezas (6 tipos × 2 colores)
- **12-13**: Turno actual
- **14-17**: Derechos de enroque
- **18-117**: Historial de jugadas (8 posiciones previas)
- **118**: Regla de 50 movimientos

### MCTS Mejorado

A diferencia del MCTS puro, el MCTS de AlphaZero:

1. **Usa probabilidades de la red** en lugar de uniformes
2. **Evalúa con la red** en lugar de simulaciones aleatorias
3. **Balance UCB mejorado** con prior de la red:

```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * √(N(s)) / (1 + N(s,a))
```

Donde:
- `Q(s,a)`: Valor medio acumulado
- `P(s,a)`: Probabilidad a priori de la red
- `N(s)`: Visitas al nodo padre
- `N(s,a)`: Visitas a este nodo

## 📈 Mejoras Futuras

- [ ] **Representación mejorada**: Historial completo de 8 jugadas
- [ ] **Mapeo exacto de movimientos**: 73 planos por movimiento
- [ ] **Augmentación de datos**: Simetría del tablero
- [ ] **Evaluación competitiva**: vs Stockfish, Minimax, MCTS
- [ ] **Entrenamiento distribuido**: Múltiples workers de auto-juego
- [ ] **Curriculum learning**: Comenzar con problemas simples
- [ ] **Transferencia de aprendizaje**: Pre-entrenar con partidas históricas

## 🔬 Experimentos Sugeridos

### 1. Comparación de Temperaturas
```bash
# Jugar con diferentes temperaturas y observar exploración
python ejemplo_alphazero.py
```

### 2. Análisis de Convergencia
```bash
# Entrenar y monitorear pérdidas
python principal.py --entrenar alphazero --num-iteraciones 50 --verbose
```

### 3. Evaluación vs Otros Agentes
```bash
# Después de entrenar, evaluar contra Minimax
python principal.py --evaluar
```

## 📚 Referencias

- **Paper Original**: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)
- **AlphaGo Zero**: Precursor que usa el mismo principio
- **AlphaZero**: Generalización a múltiples juegos (ajedrez, shogi, go)

## 🆘 Solución de Problemas

### GPU no detectada
```python
# El agente automáticamente usa CPU si CUDA no está disponible
agente = AgenteAlphaZero(dispositivo='cpu')
```

### Memoria insuficiente
```python
# Reducir tamaño de batch y número de simulaciones
python principal.py --entrenar alphazero \
    --batch-size 32 \
    --num-simulaciones 400
```

### Entrenamiento lento
- Usar GPU si está disponible
- Reducir número de episodios por iteración
- Aumentar frecuencia de guardado para checkpoints

## 📝 Notas de Desarrollo

Esta implementación es una **versión simplificada** de AlphaZero, optimizada para:
- Entrenamiento en recursos limitados
- Demostración educativa
- Base para experimentación

Para uso competitivo profesional, considerar:
- Aumentar capacidad de red (más bloques residuales)
- Entrenar por más iteraciones (miles)
- Usar múltiples GPUs
- Implementar representación completa del tablero
