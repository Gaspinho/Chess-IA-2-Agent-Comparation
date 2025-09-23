# Chess-IA-2-Agent-Comparation 🏁♟️

**Sistema de entrenamiento y evaluación comparativa de agentes de ajedrez con inteligencia artificial**

Este proyecto implementa cuatro tipos diferentes de agentes de ajedrez y proporciona un sistema completo para entrenarlos, evaluarlos y comparar sus desempeños:

- 🤖 **Minimax** - Búsqueda con poda alfa-beta
- 🎯 **MCTS** - Monte Carlo Tree Search  
- 🧠 **DQN** - Deep Q-Network (Aprendizaje por Refuerzo)
- 🚀 **PPO** - Proximal Policy Optimization (Aprendizaje por Refuerzo)

## 🚀 Características

- ✅ **Cuatro agentes implementados** con diferentes enfoques algorítmicos
- ✅ **Entorno de ajedrez** basado en PettingZoo
- ✅ **Sistema de entrenamiento** para agentes de RL
- ✅ **Evaluación automática** con enfrentamientos entre agentes
- ✅ **Análisis de desempeño** con métricas detalladas
- ✅ **Visualizaciones** y reportes automáticos
- ✅ **Interfaz de línea de comandos** fácil de usar

## 📁 Estructura del Proyecto

```
Chess-IA-2-Agent-Comparation/
├── agentes/                 # Implementaciones de agentes
│   ├── agente_minimax.py   # Minimax con poda alfa-beta
│   ├── agente_mcts.py      # Monte Carlo Tree Search
│   ├── agente_dqn.py       # Deep Q-Network
│   └── agente_ppo.py       # Proximal Policy Optimization
├── entorno/                # Wrapper del entorno de ajedrez
│   └── entorno_ajedrez.py  # Interfaz unificada con PettingZoo
├── entrenamiento/          # Scripts de entrenamiento y evaluación
│   ├── entrenar_dqn.py     # Entrenamiento DQN
│   ├── entrenar_ppo.py     # Entrenamiento PPO
│   └── evaluar_agentes.py  # Sistema de evaluación
├── analisis/               # Análisis y reportes
│   ├── metricas_desempeno.py
│   ├── visualizacion.py
│   └── generar_reporte.py
├── modelos/                # Modelos entrenados
├── resultados/             # Datos y gráficos de evaluación
├── principal.py            # Interfaz principal de línea de comandos
├── requerimientos.txt      # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd Chess-IA-2-Agent-Comparation
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requerimientos.txt
```

### Dependencias principales:
- `pettingzoo[classic]` - Entorno de ajedrez
- `stable-baselines3` - Algoritmos de RL
- `torch` - Deep Learning
- `numpy`, `pandas` - Procesamiento de datos
- `matplotlib`, `seaborn` - Visualizaciones
- `chess` - Lógica de ajedrez

## 🎮 Uso

### Interfaz de Línea de Comandos

El proyecto incluye un script principal que permite ejecutar todas las funcionalidades:

```bash
python principal.py [opciones]
```

### Entrenar Agentes

```bash
# Entrenar DQN
python principal.py --entrenar dqn --timesteps 100000

# Entrenar PPO
python principal.py --entrenar ppo --timesteps 100000 --learning-rate 0.0003

# Entrenar con parámetros personalizados
python principal.py --entrenar dqn --timesteps 50000 --batch-size 128 --gamma 0.95
```

### Evaluar Agentes

```bash
# Evaluación completa (todos vs todos)
python principal.py --evaluar

# Evaluación con parámetros personalizados
python principal.py --evaluar --timeout 15 --max-moves 300
```

### Generar Reportes

```bash
# Generar reporte de análisis
python principal.py --reporte
```

### Flujo Completo

```bash
# Entrenar ambos agentes de RL, evaluar todos y generar reporte
python principal.py --entrenar dqn --entrenar ppo --evaluar --reporte
```

## 📊 Tipos de Agentes

### 1. 🤖 Minimax
- **Algoritmo:** Búsqueda minimax con poda alfa-beta
- **Características:** Evaluación heurística, profundidad configurable
- **Fortalezas:** Jugadas tácticas sólidas, predecible
- **Parámetros:** `profundidad_maxima`, `tiempo_limite`

### 2. 🎯 MCTS (Monte Carlo Tree Search)
- **Algoritmo:** Simulaciones Monte Carlo con selección UCB1
- **Características:** Exploración balanceada, simulaciones aleatorias
- **Fortalezas:** Bueno en posiciones complejas, adaptable
- **Parámetros:** `num_simulaciones`, `c_exploracion`

### 3. 🧠 DQN (Deep Q-Network)
- **Algoritmo:** Aprendizaje por refuerzo con redes neuronales
- **Características:** Aprende de experiencia, replay buffer
- **Fortalezas:** Mejora con entrenamiento, maneja espacios grandes
- **Parámetros:** `learning_rate`, `buffer_size`, `batch_size`

### 4. 🚀 PPO (Proximal Policy Optimization)
- **Algoritmo:** Optimización de política con clipping
- **Características:** Estable, eficiente en muestras
- **Fortalezas:** Convergencia estable, buen rendimiento general
- **Parámetros:** `learning_rate`, `n_steps`, `n_epochs`

## 📈 Métricas de Evaluación

El sistema evalúa los agentes usando las siguientes métricas:

- **Tasa de victoria** - Porcentaje de partidas ganadas
- **Tasa de empate** - Porcentaje de partidas empatadas  
- **Tiempo promedio por jugada** - Eficiencia computacional
- **Recompensa promedio** - Valor estimado de las jugadas
- **Matriz de enfrentamientos** - Resultados detallados por pares

## 🎯 Ejemplos de Uso

### Ejemplo 1: Entrenamiento Rápido
```bash
# Entrenamiento rápido para pruebas
python principal.py --entrenar dqn --timesteps 10000
python principal.py --entrenar ppo --timesteps 10000
python principal.py --evaluar
```

### Ejemplo 2: Entrenamiento Completo
```bash
# Entrenamiento completo con buenos parámetros
python principal.py --entrenar dqn --timesteps 200000 --learning-rate 0.0001
python principal.py --entrenar ppo --timesteps 200000 --learning-rate 0.0003
```

### Ejemplo 3: Solo Evaluación
```bash
# Evaluar agentes pre-entrenados (si existen modelos)
python principal.py --evaluar --reporte
```

## 📊 Resultados Esperados

Después de ejecutar la evaluación completa, obtendrás:

1. **Archivo CSV** (`resultados/comparacion_agentes.csv`) con datos detallados
2. **Gráficos** en formato PNG con visualizaciones
3. **Reporte automático** (`analisis/reporte.md`) con análisis completo

### Estructura de Resultados:
```
resultados/
├── comparacion_agentes.csv      # Datos de todas las partidas
├── victorias_por_agente.png     # Gráfico de victorias
├── matriz_enfrentamientos.png   # Matriz de resultados
└── tiempo_por_jugada.png        # Tiempos de respuesta
```

## ⚙️ Configuración Avanzada

### Parámetros de Entrenamiento

#### DQN:
- `--learning-rate`: Tasa de aprendizaje (default: 1e-4)
- `--buffer-size`: Tamaño del buffer (default: 50000)
- `--batch-size`: Tamaño del lote (default: 64)
- `--gamma`: Factor de descuento (default: 0.99)

#### PPO:
- `--learning-rate`: Tasa de aprendizaje (default: 3e-4)
- `--n-steps`: Pasos por actualización (default: 2048)
- `--n-epochs`: Épocas de optimización (default: 10)
- `--batch-size`: Tamaño del lote (default: 64)

### Parámetros de Evaluación

- `--timeout`: Tiempo límite por jugada en segundos (default: 10.0)
- `--max-moves`: Máximo de jugadas por partida (default: 500)

## 🐛 Resolución de Problemas

### Problema: Error de importación de PettingZoo
```bash
pip install 'pettingzoo[classic]'
```

### Problema: Error de memoria con PyTorch
- Reducir `batch_size` y `buffer_size`
- Usar CPU en lugar de GPU: agregar `--device cpu`

### Problema: Modelos no encontrados
- Asegúrate de entrenar los agentes antes de evaluar
- Verifica que los archivos estén en `modelos/dqn_modelo.zip` y `modelos/ppo_modelo.zip`

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **PettingZoo** - Por el excelente entorno de ajedrez
- **Stable-Baselines3** - Por las implementaciones de RL
- **Python-Chess** - Por la lógica de ajedrez

## 📞 Contacto

Para preguntas, sugerencias o reportar problemas, puedes:
- Abrir un issue en GitHub
- Contactar al equipo de desarrollo

---

**¡Disfruta experimentando con diferentes agentes de ajedrez! ♟️🤖**