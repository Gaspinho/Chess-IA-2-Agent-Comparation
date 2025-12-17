# ⚡ Prueba Rápida de Stockfish

Este documento te guía para probar la implementación de Stockfish en menos de 5 minutos.

## 📋 Pasos Rápidos

### 1. Verificar Instalación (30 segundos)

```bash
python verificar_stockfish.py
```

**Si funciona**: Verás ✅ VERIFICACIÓN EXITOSA

**Si falla**: Instala Stockfish primero (ver abajo)

### 2. Ejemplo Simple (1 minuto)

```bash
python ejemplo_stockfish.py
```

Este script muestra:
- Evaluación de posiciones
- Sugerencia de movimientos
- Comparación entre niveles

### 3. Primera Partida (2-3 minutos)

```bash
python ejecutar_stockfish.py --nivel-stockfish 5 --profundidad-minimax 2
```

Partida rápida: Stockfish nivel 5 vs Minimax profundidad 2

## 🔧 Instalación Rápida de Stockfish

### Windows
1. Descarga: https://stockfishchess.org/download/
2. Extrae `stockfish.exe`
3. Opción A: Copia a `C:\Windows\System32\`
4. Opción B: Usa `--ruta-stockfish "C:\ruta\a\stockfish.exe"`

### Linux
```bash
sudo apt-get install stockfish
```

### macOS
```bash
brew install stockfish
```

## 🎯 Comandos de Prueba

### Rápido (3 minutos)
```bash
python ejecutar_stockfish.py --nivel-stockfish 5 --tiempo-stockfish 0.3 --profundidad-minimax 2
```

### Equilibrado (10 minutos)
```bash
python ejecutar_stockfish.py --nivel-stockfish 10 --simulaciones-mcts 800 --blancas stockfish --negras mcts
```

### Fuerte (20+ minutos)
```bash
python ejecutar_stockfish.py --nivel-stockfish 18 --profundidad-minimax 5 --tiempo-minimax 10.0
```

## ✅ Checklist de Funcionalidad

Después de instalar, verifica:

- [ ] `python verificar_stockfish.py` ✅
- [ ] `python ejemplo_stockfish.py` ejecuta sin errores
- [ ] `python ejecutar_stockfish.py --nivel-stockfish 5` completa una partida
- [ ] Puedes ver estadísticas al final de cada partida

## 🆘 Solución Rápida de Problemas

### "No se encontró Stockfish"
```bash
# Verifica que está instalado
where stockfish      # Windows
which stockfish      # Linux/macOS

# Si está instalado pero no se encuentra, usa ruta completa
python ejecutar_stockfish.py --ruta-stockfish "/ruta/completa/a/stockfish"
```

### "UCI initialization failed"
- Descarga la versión más reciente de Stockfish
- Asegúrate de que el ejecutable no esté corrupto

### Muy lento
- Reduce el nivel: `--nivel-stockfish 5`
- Reduce el tiempo: `--tiempo-stockfish 0.5`

## 📚 Siguiente Paso

Una vez que todo funcione, explora:

1. **STOCKFISH_GUIA.md** - Guía completa
2. **Diferentes niveles** - Prueba niveles 1-20
3. **Diferentes oponentes** - vs Minimax, vs MCTS
4. **Configuraciones personalizadas** - Ajusta todos los parámetros

## 💡 Ejemplos Rápidos por Nivel de Interés

### Solo quiero ver que funciona (1 min)
```bash
python ejemplo_stockfish.py
```

### Quiero ver una partida (3 min)
```bash
python ejecutar_stockfish.py --nivel-stockfish 5 --profundidad-minimax 2
```

### Quiero probar diferentes configuraciones (10 min)
```bash
# Stockfish débil
python ejecutar_stockfish.py --nivel-stockfish 3

# Stockfish medio
python ejecutar_stockfish.py --nivel-stockfish 10

# Stockfish fuerte
python ejecutar_stockfish.py --nivel-stockfish 18
```

### Quiero comparaciones serias (30+ min)
```bash
# Stockfish vs MCTS
python ejecutar_stockfish.py --blancas stockfish --negras mcts --nivel-stockfish 15

# MCTS vs Stockfish
python ejecutar_stockfish.py --blancas mcts --negras stockfish --nivel-stockfish 15

# Minimax vs Stockfish
python ejecutar_stockfish.py --blancas minimax --negras stockfish --nivel-stockfish 15
```

## 🎉 ¡Listo!

Si llegaste aquí y todo funciona, ¡felicidades! Ya tienes Stockfish integrado en tu proyecto de IA de ajedrez.

Para más información detallada, consulta **STOCKFISH_GUIA.md**.
