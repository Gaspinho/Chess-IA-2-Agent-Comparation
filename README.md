# Análisis Comparativo de Agentes de Ajedrez IA

Un sistema completo para entrenar, evaluar y comparar diferentes tipos de agentes de ajedrez utilizando inteligencia artificial.

## 🎯 Descripción del Proyecto

Este proyecto implementa y compara cuatro tipos diferentes de agentes de ajedrez:

- **2 Agentes de Búsqueda:**
  - **Minimax** con poda alfa-beta
  - **Monte Carlo Tree Search (MCTS)**

- **2 Agentes de Aprendizaje por Refuerzo:**
  - **Deep Q-Network (DQN)**
  - **Proximal Policy Optimization (PPO)**

El sistema permite entrenar los agentes RL, evaluar todos los agentes en torneos todos-contra-todos, y generar reportes detallados con análisis estadístico y visualizaciones.

## 📁 Estructura del Proyecto

```
Chess-IA-2-Agent-Comparation/
├── agentes/
│   ├── agente_minimax.py      # Agente Minimax con poda alfa-beta
│   ├── agente_mcts.py         # Agente Monte Carlo Tree Search
│   ├── agente_dqn.py          # Agente Deep Q-Network
│   └── agente_ppo.py          # Agente Proximal Policy Optimization
├── entorno/
│   └── entorno_ajedrez.py     # Wrapper del entorno PettingZoo Chess
├── entrenamiento/
│   ├── entrenar_dqn.py        # Script de entrenamiento DQN
│   ├── entrenar_ppo.py        # Script de entrenamiento PPO
│   └── evaluar_agentes.py     # Sistema de evaluación y torneos
├── analisis/
│   ├── metricas_desempeno.py  # Cálculo de métricas estadísticas
│   ├── visualizacion.py       # Generación de gráficos
│   └── generar_reporte.py     # Generador de reportes en Markdown
├── modelos/                   # Modelos entrenados
├── resultados/                # Resultados de evaluaciones y gráficos
├── principal.py               # Archivo principal del sistema
├── requerimientos.txt         # Dependencias del proyecto
└── LEEME.md                   # Este archivo
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Gaspinho/Chess-IA-2-Agent-Comparation.git
cd Chess-IA-2-Agent-Comparation
```

### 2. Crear entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requerimientos.txt
```

### 4. Verificar instalación
```bash
python principal.py --help
```

## 🎮 Uso del Sistema

### Comandos Básicos

El sistema se controla desde el archivo `principal.py` con diferentes opciones:

```bash
# Ver ayuda completa
python principal.py --help

# Entrenar agentes
python principal.py --entrenar dqn --timesteps 50000
python principal.py --entrenar ppo --timesteps 100000

# Evaluar agentes
python principal.py --evaluar --partidas 20

# Generar reporte
python principal.py --reporte --graficos

# Pipeline completo
python principal.py --evaluar --reporte --partidas 15 --graficos
```

### 1. Entrenar Agentes de Aprendizaje por Refuerzo

#### Entrenar DQN
```bash
# Entrenamiento básico
python principal.py --entrenar dqn --timesteps 100000

# Con parámetros personalizados
python principal.py --entrenar dqn --timesteps 200000 \
    --learning_rate 0.0001 --buffer_size 100000 --batch_size 128
```

#### Entrenar PPO
```bash
# Entrenamiento básico
python principal.py --entrenar ppo --timesteps 150000

# Con parámetros personalizados
python principal.py --entrenar ppo --timesteps 300000 \
    --learning_rate 0.0003 --n_steps 4096 --n_epochs 20
```

### 2. Evaluar Agentes

```bash
# Evaluación básica (10 partidas por enfrentamiento)
python principal.py --evaluar

# Evaluación extendida
python principal.py --evaluar --partidas 25 --tiempo_limite 10.0

# Con modelos específicos
python principal.py --evaluar --partidas 15 \
    --dqn_modelo modelos/mi_dqn.zip \
    --ppo_modelo modelos/mi_ppo.zip
```

### 3. Generar Reportes

```bash
# Reporte básico
python principal.py --reporte

# Reporte con gráficos
python principal.py --reporte --graficos

# Usando resultados específicos
python principal.py --reporte --graficos \
    --resultados resultados/mi_evaluacion.csv \
    --output_reporte analisis/mi_reporte.md
```

### 4. Pipeline Completo

```bash
# Entrenar, evaluar y generar reporte
python principal.py --entrenar dqn --timesteps 100000
python principal.py --entrenar ppo --timesteps 100000
python principal.py --evaluar --partidas 20
python principal.py --reporte --graficos
```

## 🧠 Descripción de los Agentes

### 1. Minimax (Agente de Búsqueda)
- **Algoritmo:** Minimax con poda alfa-beta
- **Profundidad:** 3 niveles (configurable)
- **Evaluación:** Material + posición + movilidad + seguridad del rey
- **Tiempo:** ~2-3 segundos por movimiento

```python
# Configuración en evaluación
--minimax_profundidad 4  # Mayor profundidad = mejor juego, más lento
```

### 2. MCTS (Agente de Búsqueda)
- **Algoritmo:** Monte Carlo Tree Search
- **Simulaciones:** 1000 por movimiento (configurable)
- **Exploración:** UCB1 con C=1.4
- **Política:** Priorización de capturas y jaques

```python
# Configuración en evaluación
--mcts_simulaciones 2000  # Más simulaciones = mejor juego, más lento
```

### 3. DQN (Aprendizaje por Refuerzo)
- **Red:** CNN para procesar tablero 8x8x12
- **Espacio de acción:** 4096 movimientos posibles
- **Experiencia:** Buffer de 50,000 transiciones
- **Exploración:** ε-greedy decreciente

**Parámetros de entrenamiento:**
```python
--learning_rate 0.0001      # Tasa de aprendizaje
--buffer_size 50000         # Tamaño del buffer de experiencia
--batch_size 64             # Tamaño del batch
--gamma 0.99                # Factor de descuento
--target_update_interval 1000  # Actualización de red objetivo
```

### 4. PPO (Aprendizaje por Refuerzo)
- **Red:** CNN similar a DQN
- **Política:** Actor-Critic
- **Ventaja:** Generalized Advantage Estimation (GAE)
- **Clipping:** Previene actualizaciones grandes

**Parámetros de entrenamiento:**
```python
--n_steps 2048              # Pasos por actualización
--n_epochs 10               # Épocas por actualización
--learning_rate 0.0003      # Tasa de aprendizaje
--clip_range 0.2            # Rango de clipping PPO
--gae_lambda 0.95           # Factor lambda GAE
```

## 📊 Sistema de Evaluación

### Métricas Calculadas

1. **Básicas:**
   - Victorias, empates, derrotas
   - Tasa de victorias
   - Puntuación (V + 0.5×E)
   - Tiempo promedio por movimiento

2. **Avanzadas:**
   - Rating ELO dinámico
   - Consistencia temporal
   - Eficiencia por color (blancas/negras)
   - Análisis de terminaciones
   - Matriz de enfrentamientos

3. **Estadísticas:**
   - Distribución de movimientos por partida
   - Formas de terminación
   - Performance contra diferentes oponentes

### Formato de Torneo

- **Todos contra todos:** Cada agente juega contra todos los demás
- **Alternancia de colores:** Garantiza equidad
- **Múltiples rondas:** Configurable (defecto: 10 partidas por enfrentamiento)
- **Límite de tiempo:** 5 segundos por movimiento (configurable)

## 📈 Reportes y Visualizaciones

### Gráficos Generados

1. **Resultados por Agente:** Barras de V/E/D
2. **Tasas de Rendimiento:** Barras apiladas con porcentajes
3. **Tiempos Promedio:** Comparación de velocidad
4. **Matriz de Enfrentamientos:** Heatmap de éxito
5. **Evolución ELO:** Progresión durante el torneo
6. **Distribución de Movimientos:** Histogramas por agente
7. **Radar de Métricas:** Perfil multidimensional
8. **Dashboard Completo:** Resumen visual integral

### Tipos de Reporte

1. **Reporte Completo (`reporte_completo.md`):**
   - Análisis detallado por agente
   - Estadísticas completas
   - Visualizaciones integradas
   - Conclusiones y recomendaciones

2. **Reporte Rápido (`reporte_rapido.md`):**
   - Resumen ejecutivo
   - Ranking final
   - Métricas clave

## 🔧 Configuración Avanzada

### Parámetros del Entorno

```python
# En el código, modificar entorno_ajedrez.py
tiempo_limite_movimiento = 10.0  # Segundos por movimiento
max_movimientos_partida = 300    # Límite de movimientos
```

### Personalizar Agentes

```python
# Modificar parámetros de Minimax
profundidad_maxima = 4
tiempo_limite = 8.0

# Modificar parámetros de MCTS
num_simulaciones = 2000
c_param = 1.4  # Exploración UCB1
```

### Entrenamiento Personalizado

```python
# Entrenar DQN con configuración específica
python principal.py --entrenar dqn \
    --timesteps 500000 \
    --learning_rate 0.00005 \
    --buffer_size 100000 \
    --batch_size 128 \
    --gamma 0.995 \
    --exploration_fraction 0.2
```

## 📋 Ejemplos de Uso Completos

### Ejemplo 1: Evaluación Rápida
```bash
# Solo agentes de búsqueda (no requiere entrenamiento)
python principal.py --evaluar --partidas 10 --reporte --graficos
```

### Ejemplo 2: Entrenamiento y Evaluación Completa
```bash
# 1. Entrenar agentes RL
python principal.py --entrenar dqn --timesteps 100000
python principal.py --entrenar ppo --timesteps 150000

# 2. Evaluar todos los agentes
python principal.py --evaluar --partidas 20 --tiempo_limite 8.0

# 3. Generar reporte completo
python principal.py --reporte --graficos
```

### Ejemplo 3: Comparación de Configuraciones
```bash
# Entrenar DQN con diferentes configuraciones
python principal.py --entrenar dqn --timesteps 200000 --learning_rate 0.0001
python principal.py --entrenar dqn --timesteps 200000 --learning_rate 0.0005

# Evaluar con modelos específicos
python principal.py --evaluar --partidas 15 \
    --dqn_modelo modelos/dqn_lr_0001.zip
```

## 🛠️ Resolución de Problemas

### Problemas Comunes

1. **Error de importación de PettingZoo:**
   ```bash
   pip install --upgrade pettingzoo[classic]
   ```

2. **Error de PyTorch/CUDA:**
   ```bash
   # Solo CPU
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # Con CUDA (reemplazar cu118 con su versión)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memoria insuficiente durante entrenamiento:**
   ```bash
   # Reducir batch size y buffer size
   python principal.py --entrenar dqn --batch_size 32 --buffer_size 25000
   ```

4. **Entrenamiento muy lento:**
   ```bash
   # Reducir timesteps inicialmente
   python principal.py --entrenar ppo --timesteps 50000
   ```

### Logs y Depuración

```bash
# Activar logging detallado
python principal.py --evaluar --log_level DEBUG

# Ver logs en archivo
# Los logs se guardan automáticamente con timestamp
```

## 📚 Recursos Adicionales

### Documentación de Dependencias
- [PettingZoo Chess](https://pettingzoo.farama.org/environments/classic/chess/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Python Chess](https://python-chess.readthedocs.io/)

### Papers de Referencia
- **Minimax:** Shannon, C. (1950) - Programming a Computer for Playing Chess
- **MCTS:** Coulom, R. (2006) - Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
- **DQN:** Mnih, V. et al. (2015) - Human-level control through deep reinforcement learning
- **PPO:** Schulman, J. et al. (2017) - Proximal Policy Optimization Algorithms

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar cambios con pruebas
4. Documentar modificaciones
5. Crear Pull Request

### Áreas de Mejora Sugeridas

- Implementar más algoritmos (AlphaZero, Stockfish integration)
- Mejorar funciones de evaluación de búsqueda
- Optimizar entrenamiento de agentes RL
- Agregar más métricas de análisis
- Implementar modo de juego interactivo

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE para detalles.

## 👨‍💻 Autor

**Gaspar** - [GitHub](https://github.com/Gaspinho)

---

## 🎯 Próximos Pasos Recomendados

1. **Primera ejecución:** Ejecutar solo agentes de búsqueda
   ```bash
   python principal.py --evaluar --partidas 5 --reporte --graficos
   ```

2. **Entrenar un agente RL:** Comenzar con timesteps bajos
   ```bash
   python principal.py --entrenar dqn --timesteps 50000
   ```

3. **Evaluación completa:** Una vez entrenados ambos agentes RL
   ```bash
   python principal.py --evaluar --partidas 15 --reporte --graficos
   ```

4. **Optimización:** Experimentar con diferentes parámetros y configuraciones

¡Disfruta explorando el fascinante mundo de la IA en ajedrez! 🏆♟️
