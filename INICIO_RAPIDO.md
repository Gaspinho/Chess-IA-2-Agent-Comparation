# 🚀 INICIO RÁPIDO - AlphaZero

## ¿Qué se hizo?

Se **reemplazó DQN por AlphaZero**, un algoritmo mucho más avanzado y apropiado para ajedrez que combina:
- 🎯 Monte Carlo Tree Search (MCTS)
- 🧠 Redes Neuronales Profundas
- 🔄 Auto-juego para aprendizaje

## ⚡ Empezar en 3 pasos

### 1️⃣ Instalar dependencias
```bash
pip install -r requerimientos.txt
```

### 2️⃣ Ver ejemplos
```bash
python ejemplo_alphazero.py
```

### 3️⃣ Entrenar tu primer modelo
```bash
# Entrenamiento de prueba (10 minutos)
python principal.py --entrenar alphazero --num-iteraciones 5 --num-episodios 10
```

## 📁 Archivos Nuevos Importantes

| Archivo | Descripción |
|---------|-------------|
| `agentes/agente_alphazero.py` | ⭐ Implementación completa de AlphaZero |
| `entrenamiento/entrenar_alphazero.py` | 🎓 Sistema de auto-juego y entrenamiento |
| `ejemplo_alphazero.py` | 📖 Ejemplos prácticos de uso |
| `ALPHAZERO_INFO.md` | 📚 Documentación técnica completa |
| `INSTALACION_Y_VERIFICACION.md` | 🔧 Guía de solución de problemas |
| `CAMBIOS_DQN_A_ALPHAZERO.md` | 📋 Resumen detallado de cambios |

## 🎮 Comandos Útiles

### Ver ayuda
```bash
python principal.py --help
```

### Entrenamiento rápido (prueba)
```bash
python principal.py --entrenar alphazero --num-iteraciones 10
```

### Entrenamiento completo (recomendado)
```bash
python principal.py --entrenar alphazero --num-iteraciones 100
```

### Entrenamiento personalizado
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 200 \
    --num-simulaciones 800 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --temperatura 1.0
```

### Jugar Minimax vs MCTS (sin AlphaZero)
```bash
python ejecutar_minimax_vs_mcts.py
```

### Ver AlphaZero Jugar 🎮
```bash
# Script interactivo para ver partidas en tiempo real
python ver_alphazero_jugar.py

# Opciones:
# - AlphaZero vs Minimax
# - AlphaZero vs MCTS
# - AlphaZero vs AlphaZero
# Y más...
```

**Controles:** ESC (salir), SPACE (pausar)

## 📊 ¿Por qué AlphaZero es mejor que DQN?

| Aspecto | DQN | AlphaZero |
|---------|-----|-----------|
| **Para ajedrez** | ❌ No diseñado | ✅ Perfecto |
| **Planificación** | ❌ No | ✅ MCTS |
| **Auto-mejora** | ❌ Necesita oponentes | ✅ Auto-juego |
| **Calidad** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 Próximos Pasos

1. ✅ **Instalar dependencias**: `pip install -r requerimientos.txt`
2. ✅ **Ver ejemplos**: `python ejemplo_alphazero.py`
3. ✅ **Entrenamiento de prueba**: 5-10 iteraciones
4. ⬜ **Entrenamiento completo**: 100+ iteraciones
5. ⬜ **Evaluar vs otros agentes**: (próximamente)

## 📖 Documentación

- **🚀 Este archivo**: Inicio rápido
- **📚 ALPHAZERO_INFO.md**: Documentación técnica completa
- **🔧 INSTALACION_Y_VERIFICACION.md**: Guía de instalación y troubleshooting
- **📋 CAMBIOS_DQN_A_ALPHAZERO.md**: Resumen detallado de todos los cambios
- **📖 README.md**: Guía general del proyecto

## 💡 Ejemplos de Uso

### Cargar modelo entrenado
```python
from agentes.agente_alphazero import AgenteAlphaZero

agente = AgenteAlphaZero(
    modelo_path='modelos/alphazero_final.pth',
    num_simulaciones=800
)
```

### Jugar con el agente
```python
# Seleccionar mejor jugada
accion = agente.seleccionar_accion(observacion, acciones_legales, info)

# Ver estadísticas
stats = agente.obtener_estadisticas()
print(stats)
```

## ⚙️ Configuraciones Recomendadas

### 💻 Para CPU (desarrollo)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 20 \
    --num-simulaciones 400 \
    --batch-size 32
```
⏱️ Tiempo: ~1 hora

### 🎮 Para GPU (entrenamiento serio)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 100 \
    --num-simulaciones 800 \
    --batch-size 64
```
⏱️ Tiempo: ~2-3 horas

### 🚀 Para GPU potente (investigación)
```bash
python principal.py --entrenar alphazero \
    --num-iteraciones 500 \
    --num-simulaciones 1600 \
    --batch-size 128
```
⏱️ Tiempo: ~8-12 horas

## 🆘 ¿Problemas?

### Error de imports
```bash
pip install torch chess numpy pandas matplotlib seaborn pygame
```

### Sin GPU / CUDA
No hay problema! AlphaZero funciona en CPU, solo será más lento.

### Memoria insuficiente
```bash
# Reducir batch size
python principal.py --entrenar alphazero --batch-size 16
```

### Más ayuda
Ver `INSTALACION_Y_VERIFICACION.md` para solución completa de problemas.

## ✨ Resumen

Has reemplazado exitosamente DQN por AlphaZero, obteniendo:

✅ Algoritmo estado del arte para ajedrez  
✅ Sistema de auto-juego implementado  
✅ Arquitectura de red neuronal avanzada  
✅ Documentación completa  
✅ Ejemplos prácticos  

**¡Listo para entrenar tu propio agente de ajedrez de nivel maestro!** 🏆

---

**Siguiente paso**: `python ejemplo_alphazero.py`
