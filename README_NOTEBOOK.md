# Chess AI Comparison - Jupyter Notebook Edition

## 🎯 Descripción del Proyecto

Este proyecto compara tres agentes de inteligencia artificial para ajedrez:
- **Minimax**: Algoritmo de búsqueda en árbol con poda alfa-beta
- **MCTS (Monte Carlo Tree Search)**: Búsqueda basada en simulaciones
- **PPO (Proximal Policy Optimization)**: Aprendizaje profundo por refuerzo

## 📊 Características Principales

### ✨ Nuevo en la Versión Notebook
- **Jupyter Notebook Interactivo**: Análisis y visualización paso a paso
- **Integración con Lichess**: Visualización de partidas en formato SVG y enlaces directos a Lichess
- **Gráficos Comparativos Avanzados**:
  - Matriz de win rate entre agentes
  - Distribución de victorias/empates/derrotas
  - Tiempo promedio por partida
  - Número de jugadas promedio
  - Razones de término de partidas
- **Agente PPO**: Nuevo agente de aprendizaje profundo por refuerzo
- **Análisis Detallado**: Estadísticas completas de cada agente

## 🚀 Instalación

### Requisitos Previos
- Python 3.8+
- Jupyter Notebook o JupyterLab

### Instalación de Dependencias

```bash
pip install -r requerimientos.txt
```

### Dependencias Principales
- `pettingzoo[classic]`: Entorno de ajedrez multi-agente
- `stable-baselines3`: Implementación de PPO
- `torch`: Framework de deep learning
- `chess`: Librería de ajedrez con soporte para PGN y SVG
- `matplotlib`, `seaborn`: Visualizaciones
- `pandas`: Análisis de datos
- `jupyter`: Entorno de notebooks

## 📁 Estructura del Proyecto

```
IA_PARA_JUEGOS/
├── chess_ai_comparison.ipynb    # 🆕 Notebook principal
├── agentes/
│   ├── agente_minimax.py       # Agente Minimax
│   ├── agente_mcts.py          # Agente MCTS
│   └── agente_ppo.py           # 🆕 Agente PPO
├── entorno/
│   └── entorno_ajedrez.py      # Wrapper del entorno
├── analisis/
│   ├── visualizacion.py        # Generador de gráficos
│   └── metricas_desempeno.py   # Métricas de rendimiento
├── requerimientos.txt
└── README.md
```

## 🎮 Uso

### 1. Abrir el Notebook

```bash
jupyter notebook chess_ai_comparison.ipynb
```

### 2. Ejecutar el Análisis Completo

El notebook está organizado en secciones:

1. **Instalación y Configuración**: Importa librerías y configura el entorno
2. **Entorno de Ajedrez**: Inicializa el entorno PettingZoo
3. **Agentes**: Crea instancias de Minimax, MCTS y PPO
4. **Integración con Lichess**: Herramientas de visualización
5. **Sistema de Evaluación**: Ejecuta partidas entre agentes
6. **Entrenamiento PPO**: Entrena el agente de deep learning
7. **Torneo Round-Robin**: Compara todos los agentes
8. **Visualización Lichess**: Enlaces directos a partidas
9. **Gráficos Comparativos**: Análisis visual de rendimiento
10. **Tabla de Clasificación**: Rankings finales
11. **Análisis Detallado**: Estudio de partidas específicas
12. **Exportación**: Guarda resultados en CSV

### 3. Personalización

#### Configurar Parámetros de Agentes

```python
# Minimax
minimax = AgenteMinimax(profundidad_maxima=4, tiempo_limite=10.0)

# MCTS
mcts = AgenteMCTS(num_simulaciones=2000, tiempo_limite=10.0)

# PPO
ppo_agent = crear_agente_ppo(env, learning_rate=1e-4, n_steps=4096)
```

#### Configurar Torneo

```python
evaluador = EvaluadorAgentes(max_jugadas=300, timeout_jugada=15.0)
df_resultados = evaluador.torneo_round_robin(agentes_torneo, partidas_por_pareja=5)
```

## 📈 Análisis de Resultados

### Visualizaciones Incluidas

1. **Gráfico de Victorias**: Compara W/D/L de cada agente
2. **Matriz de Win Rate**: Heatmap de rendimiento relativo
3. **Tiempo Promedio**: Eficiencia computacional
4. **Jugadas Promedio**: Duración de partidas
5. **Razones de Término**: Distribución de finales (mate, ahogado, etc.)

### Integración con Lichess

Cada partida genera:
- **PGN**: Formato estándar de notación de ajedrez
- **URL de Lichess**: Enlace directo para análisis interactivo
- **SVG del tablero**: Visualización directa en el notebook

Ejemplo de URL generada:
```
https://lichess.org/paste?pgn=...
```

### Exportación de Datos

Los resultados se exportan automáticamente:
- `resultados_torneo_YYYYMMDD_HHMMSS.csv`: Todas las partidas
- `clasificacion_YYYYMMDD_HHMMSS.csv`: Tabla de clasificación

## 🤖 Descripción de los Agentes

### Minimax (Búsqueda en Árbol)
- **Algoritmo**: Minimax con poda alfa-beta
- **Evaluación**: Función heurística basada en material y posición
- **Fortalezas**: Cálculo táctico preciso, rápido en profundidades bajas
- **Debilidades**: Limitado por profundidad de búsqueda

### MCTS (Monte Carlo Tree Search)
- **Algoritmo**: Selección UCB1 + simulaciones aleatorias
- **Evaluación**: Estimación estocástica mediante playouts
- **Fortalezas**: No requiere evaluación heurística, explora amplias posibilidades
- **Debilidades**: Requiere muchas simulaciones, puede ser lento

### PPO (Proximal Policy Optimization)
- **Algoritmo**: Actor-Critic con clipping de políticas
- **Aprendizaje**: Redes neuronales profundas entrenadas con RL
- **Fortalezas**: Aprende patrones complejos, mejora con experiencia
- **Debilidades**: Requiere entrenamiento extensivo, menos interpretable

## 🔧 Configuración Avanzada

### Entrenamiento Extendido de PPO

```python
# Entrenamiento largo para mejor rendimiento
ppo_agent.entrenar(total_timesteps=1_000_000, progreso=True)

# Guardar modelo entrenado
ppo_agent.guardar_modelo("modelos/ppo_chess_1M.zip")

# Cargar modelo previamente entrenado
ppo_agent.cargar_modelo("modelos/ppo_chess_1M.zip")
```

### Evaluación Individual

```python
# Jugar una partida específica con visualización
resultado = evaluador.jugar_partida(
    minimax, mcts, 
    "Minimax", "MCTS",
    visualizar=True  # Muestra el tablero durante la partida
)

# Ver en Lichess
print(resultado.url_lichess)
```

## 📊 Métricas de Rendimiento

El notebook calcula automáticamente:

- **Win Rate**: Porcentaje de victorias
- **Puntos**: Sistema de puntuación (W=1, D=0.5, L=0)
- **Tiempo por Jugada**: Eficiencia computacional
- **Nodos Evaluados** (Minimax): Profundidad de búsqueda
- **Simulaciones** (MCTS): Exploraciones realizadas
- **Loss** (PPO): Pérdida durante entrenamiento

## 🎓 Casos de Uso

### Para Investigadores
- Comparación rigurosa de algoritmos de búsqueda vs RL
- Análisis de trade-offs entre tiempo y calidad
- Estudio de patrones de juego emergentes

### Para Estudiantes
- Aprendizaje interactivo de algoritmos de IA
- Visualización del proceso de decisión
- Experimentación con hiperparámetros

### Para Desarrolladores
- Base para desarrollo de agentes personalizados
- Framework extensible para nuevos algoritmos
- Herramientas de evaluación robustas

## 🐛 Solución de Problemas

### Error: "Modelo no inicializado"
```python
# Asegurarse de crear el modelo antes de entrenar
ppo_agent.crear_modelo(env)
ppo_agent.entrenar(total_timesteps=50000)
```

### Error: "Acción ilegal"
Los agentes automáticamente manejan acciones ilegales, pero si persiste:
```python
# Verificar que las acciones legales estén correctamente obtenidas
acciones_legales = env.obtener_acciones_legales()
```

### Lichess no muestra la partida
Verificar que el PGN sea válido:
```python
print(resultado.pgn)
```

## 📚 Referencias

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Python Chess Library](https://python-chess.readthedocs.io/)
- [Lichess API](https://lichess.org/api)

## 🤝 Contribuciones

Las contribuciones son bienvenidas:
- Nuevos agentes (AlphaZero, MuZero, etc.)
- Mejoras en visualización
- Optimizaciones de rendimiento
- Documentación adicional

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 🙏 Agradecimientos

- PettingZoo por el entorno de ajedrez multi-agente
- Stable-Baselines3 por la implementación de PPO
- Lichess por la plataforma de análisis de ajedrez

---

**¡Disfruta explorando la inteligencia artificial en el ajedrez! ♟️🤖**
