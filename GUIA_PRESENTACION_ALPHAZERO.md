# Guía de Presentación: AlphaZero en Ajedrez

**Fecha:** Diciembre 2025  
**Proyecto:** Comparación de Agentes de IA para Ajedrez

---

## 1. Resumen Ejecutivo (30 segundos)

AlphaZero es un algoritmo de aprendizaje por refuerzo que **aprende a jugar sin datos previos** mediante:
- **Auto-juego:** El agente juega contra sí mismo
- **MCTS guiado:** Búsqueda en árbol guiada por una red neuronal
- **Mejora iterativa:** Los datos del auto-juego entrenan la red, que mejora el MCTS

**Resultado:** El sistema mejora continuamente sin conocimiento humano experto.

---

## 2. Arquitectura del Sistema

### 2.1 Red Neuronal: `RedNeuralAlphaZero`
**Ubicación:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L23-L98)

**Entrada:** 
- Tensor 119×8×8 representando el tablero
- 119 canales: posiciones de piezas + metadatos (turno, enroque, etc.)

**Arquitectura:**
```
Entrada (119×8×8)
    ↓
Capa Convolucional + BatchNorm
    ↓
10 Bloques Residuales (256 canales)
    ↓
    ├─→ Cabeza de Política (4672 movimientos posibles)
    └─→ Cabeza de Valor (escalar en [-1, 1])
```

**Salidas:**
1. **Política (π):** Distribución de probabilidad sobre movimientos
   - Log-softmax de 4672 dimensiones
   - Indica qué tan "prometedor" es cada movimiento
   
2. **Valor (v):** Evaluación de la posición
   - Escalar entre -1 (pérdida segura) y +1 (victoria segura)
   - Predice el resultado esperado del juego

**Código clave:**
```python
def forward(self, x):
    # Torre residual
    x = F.relu(self.bn_entrada(self.conv_entrada(x)))
    for bloque in self.bloques_res:
        x = bloque(x)
    
    # Política: distribución sobre movimientos
    politica = F.log_softmax(self.fc_politica(...), dim=1)
    
    # Valor: evaluación de posición
    valor = torch.tanh(self.fc_valor2(...))
    
    return politica, valor
```

### 2.2 Monte Carlo Tree Search (MCTS)
**Ubicación:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L121-L189)

**Estructura del Nodo:**
```python
class NodoMCTSAlphaZero:
    visitas: int              # Número de veces visitado
    valor_total: float        # Suma de valores propagados
    prob_previa: float        # P(a) de la red neuronal
    hijos: Dict[Move, Nodo]   # Nodos hijos por movimiento
```

**Fórmula PUCT (Predictor + Upper Confidence Tree):**

$$a^* = \arg\max_a \left( Q(s,a) + U(s,a) \right)$$

Donde:
- $Q(s,a) = \frac{W(s,a)}{N(s,a)}$ → **Explotación** (valor promedio)
- $U(s,a) = c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$ → **Exploración** (guiada por política)

**Parámetros:**
- $c_{puct}$ = 1.5 (constante de exploración)
- $P(s,a)$ = probabilidad de la red neuronal (prior)
- $N(s,a)$ = visitas al nodo hijo
- $N(s)$ = visitas al nodo padre

### 2.3 Proceso de Búsqueda MCTS (Por Jugada)

**Ubicación:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L239-L270)

**4 Fases por simulación:**

1. **Selección:** Bajar por el árbol eligiendo el hijo con mayor Q + U
   ```python
   while not nodo.es_hoja():
       movimiento, nodo = self._seleccionar_hijo(nodo)  # Maximiza PUCT
       tablero.push(movimiento)
   ```

2. **Expansión:** Si es hoja no terminal, evaluar con la red
   ```python
   politica, valor = self.red_neuronal(estado)
   nodo.expandir(probabilidades_movimientos)  # Crear hijos con P(a)
   ```

3. **Evaluación:** 
   - Si terminal: usar resultado real (+1, 0, -1)
   - Si no terminal: usar predicción de valor de la red

4. **Retropropagación:** Subir el valor cambiando signo por nivel
   ```python
   while nodo is not None:
       nodo.actualizar(valor)
       valor = -valor  # Cambio de perspectiva
       nodo = nodo.padre
   ```

**Selección final:** Tras N simulaciones, elegir el movimiento más visitado
```python
mejor_movimiento = max(raiz.hijos.items(), key=lambda x: x[1].visitas)[0]
```

---

## 3. Pipeline de Entrenamiento

**Ubicación:** [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py#L36-L124)

### 3.1 Ciclo Principal

```
Para cada iteración:
    ├─→ Fase 1: Auto-juego (100 partidas)
    │   └─→ Genera datos (s, π, z)
    │
    ├─→ Fase 2: Agregar a buffer de experiencia (máx 10,000)
    │
    ├─→ Fase 3: Entrenar red con batches del buffer
    │   ├─→ Loss de política: -Σ(π_target * log(π_pred))
    │   └─→ Loss de valor: MSE(v_pred, z_target)
    │
    └─→ Fase 4: Guardar modelo cada N iteraciones
```

### 3.2 Generación de Datos (Auto-juego)

**Código:** [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py#L150-L233)

**Por cada partida:**
1. Iniciar tablero vacío
2. Mientras no termine:
   - Ejecutar 800 simulaciones MCTS desde posición actual
   - Guardar estado $s$ (representación del tablero)
   - Extraer política $\pi$ de las visitas: $\pi_a = \frac{N_a}{\sum_b N_b}$
   - Seleccionar movimiento (con temperatura)
   - Aplicar movimiento
3. Al terminar, calcular resultado $z$:
   - +1 si ganó el jugador del turno
   - -1 si perdió
   - 0 si empate

**Datos generados:** Lista de tuplas `(estado, política_objetivo, valor_objetivo)`

### 3.3 Objetivos de Entrenamiento

**Política objetivo ($\pi_{target}$):**
- Distribución normalizada de visitas MCTS
- Representa la "mejor jugada" según búsqueda exhaustiva
```python
π_target[movimiento] = visitas[movimiento] / sum(visitas)
```

**Valor objetivo ($z_{target}$):**
- Resultado real de la partida desde perspectiva del turno
- **Mejora implementada:** Ahora usa `tablero_final.result()` correctamente
```python
if resultado == "1-0":
    z = +1.0 si turno == BLANCAS else -1.0
elif resultado == "0-1":
    z = -1.0 si turno == BLANCAS else +1.0
else:
    z = 0.0  # Empate
```

### 3.4 Funciones de Pérdida

**Código:** [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py#L303-L327)

```python
# Pérdida de política (entropía cruzada)
loss_politica = -torch.mean(torch.sum(π_target * log(π_pred), dim=1))

# Pérdida de valor (MSE)
loss_valor = MSE(v_pred, z_target)

# Pérdida total
loss_total = loss_politica + loss_valor
```

**Optimizador:** Adam con learning rate 0.001 y weight decay 1e-4

---

## 4. Mejoras Implementadas

### 4.1 ✅ Valor Objetivo Real
**Archivo:** [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py#L283-L306)

**Antes:** Retornaba valores aleatorios (-1, 0, 1)  
**Ahora:** Calcula resultado desde `tablero.result()` con perspectiva correcta

```python
def _evaluar_resultado(self, estado_final, tablero_final, turno_inicial):
    resultado = tablero_final.result()
    
    if resultado == "1-0":
        return 1.0 if turno_inicial == chess.WHITE else -1.0
    elif resultado == "0-1":
        return 1.0 if turno_inicial == chess.BLACK else -1.0
    else:
        return 0.0  # Empate
```

**Además:** Alterna perspectiva por turno en los datos de entrenamiento

### 4.2 ✅ Mapeo de Promociones
**Archivo:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L522-L566)

**Antes:** Índice simple `from * 64 + to` (no diferenciaba promociones)  
**Ahora:** Offset adicional para promociones

```python
if movimiento.promotion is not None:
    offset_promocion = 4096
    tipo_offset = {QUEEN: 0, ROOK: 1, BISHOP: 2, KNIGHT: 3}
    idx = offset_promocion + from_square * 4 + tipo_offset[promotion]
```

---

## 5. Flujo Completo de una Jugada

```
Estado del tablero (s)
    ↓
Convertir a tensor (119×8×8)
    ↓
Inicializar nodo raíz MCTS
    ↓
Repetir 800 veces:
    │
    ├─→ Selección: Bajar por Q + U
    ├─→ Expansión: Red(s) → (π, v)
    ├─→ Crear hijos con P(a) = π(a)
    └─→ Retropropagar valor
    ↓
Tras 800 simulaciones:
    ↓
Elegir movimiento por visitas
    ↓
Aplicar en tablero real
```

**Tiempo típico:** ~2-5 segundos por jugada (CPU), ~0.5s (GPU)

---

## 6. Preguntas Frecuentes del Profesor

### P1: ¿Qué predice exactamente la red neuronal?

**R:** Dos cosas simultáneamente:
1. **Política (π):** Vector de 4672 dimensiones con probabilidades de cada movimiento posible. Indica "qué jugadas son prometedoras según patrones aprendidos".
2. **Valor (v):** Un número entre -1 y +1 que estima el resultado esperado desde la posición actual (-1 = perderemos, 0 = empate, +1 = ganaremos).

**Código:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L75-L98)

---

### P2: ¿Cómo se relaciona la red neuronal con MCTS?

**R:** La red **guía y acelera** el MCTS:
- **Política (π):** Se usa como **prior** $P(a)$ en la fórmula PUCT. Aumenta la exploración de jugadas que la red considera buenas.
- **Valor (v):** Se usa para **evaluar hojas** del árbol sin simular hasta el final. Reemplaza rollouts aleatorios.

**Sin red:** MCTS exploraría uniformemente (lento e ineficiente)  
**Con red:** MCTS enfoca simulaciones en ramas prometedoras (10-100x más eficiente)

**Código:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L166-L189, agentes/agente_alphazero.py#L272-L330)

---

### P3: ¿Cómo se entrena la red? ¿Qué son los objetivos?

**R:** La red aprende por **imitación supervisada** de las jugadas del MCTS:

**Objetivo de Política:**
- Target: Distribución de visitas MCTS → $\pi_a = N_a / \sum_b N_b$
- Intuición: "Imita las jugadas que MCTS considera mejores tras búsqueda exhaustiva"
- Loss: Entropía cruzada entre predicción y visitas normalizadas

**Objetivo de Valor:**
- Target: Resultado real del juego (+1/-1/0)
- Intuición: "Aprende a predecir quién ganará desde cada posición"
- Loss: MSE entre predicción y resultado

**Código:** [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py#L303-L327)

---

### P4: ¿Qué es la temperatura y para qué sirve?

**R:** Controla **exploración vs explotación** en la selección final:

$$p_a \propto N_a^{1/T}$$

- **T = 1:** Samplea proporcionalmente a visitas (explora variedad)
- **T = 0:** Elige el más visitado determinísticamente (explota lo mejor)
- **T > 1:** Más exploratorio (jugadas poco visitadas tienen más chance)

**Uso típico:**
- Primeras 30 jugadas del juego: T = 1 (explorar aperturas)
- Resto del juego: T = 0 (jugar lo mejor)
- Evaluación/competencia: T = 0 siempre

**Código:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L351-L387)

---

### P5: ¿Por qué se cambia el signo del valor en la retropropagación?

**R:** Porque en juegos de suma cero, una buena posición para un jugador es mala para el oponente.

Si desde la posición actual el valor es +0.8 (bueno para quien juega ahora), para el jugador anterior (que nos llevó aquí) es -0.8 (malo, porque nos dio ventaja).

**Matemática:**
- Nodo hijo (mi turno): v = +0.8
- Nodo padre (turno oponente): v = -0.8
- Nodo abuelo (mi turno de nuevo): v = +0.8

**Código:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L332-L348)

```python
while nodo is not None:
    nodo.actualizar(valor)
    valor = -valor  # Cambiar perspectiva
    nodo = nodo.padre
```

---

### P6: ¿Cuál es la diferencia entre este MCTS y el MCTS tradicional?

**R:** MCTS tradicional vs AlphaZero MCTS:

| Aspecto | MCTS Tradicional | AlphaZero MCTS |
|---------|-----------------|----------------|
| **Expansión** | Uniforme o heurística simple | Guiada por red neuronal (π) |
| **Evaluación** | Rollout aleatorio hasta el final | Red neuronal (v) evalúa instantáneamente |
| **Selección** | UCB1 básico | PUCT con priors de la red |
| **Velocidad** | Miles de simulaciones lentas | Cientos de simulaciones profundas |
| **Conocimiento** | Minimal | Aprende patrones estratégicos |

**Ventaja clave:** La red neuronal "resume" el conocimiento de millones de juegos previos.

---

### P7: ¿Cómo se representa el tablero para la red neuronal?

**R:** Tensor 119×8×8 con múltiples "planos":

**Canales 0-5:** Piezas blancas (peones, caballos, alfiles, torres, damas, rey)  
**Canales 6-11:** Piezas negras  
**Canal 12:** Turno actual = blancas  
**Canal 13:** Turno actual = negras  
**Canales 14-17:** Derechos de enroque (K/Q para ambos colores)  
**Canales 18+:** Repeticiones, regla 50 movimientos, etc.

Cada celda [i,j] del plano es 1.0 si la pieza está ahí, 0.0 si no.

**Código:** [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py#L468-L520)

---

### P8: ¿Cuánto tiempo toma entrenar AlphaZero desde cero?

**R:** En este proyecto (configuración reducida):

- **Por iteración:** ~10-30 minutos (depende de CPU/GPU)
  - 100 partidas de auto-juego: ~20 min
  - Entrenamiento de batches: ~5 min
- **Total para jugar decentemente:** 50-100 iteraciones (~1-2 días CPU)
- **Nivel experto:** 1000+ iteraciones (~semanas)

**AlphaZero original (DeepMind):**
- 44 millones de partidas
- 9 horas en 5000 TPUs
- Superó a Stockfish

**Optimizaciones posibles:**
- Usar GPU (10-50x más rápido)
- Paralelizar auto-juego
- Reducir simulaciones en primeras iteraciones

---

### P9: ¿Cuáles son las limitaciones de esta implementación?

**R:** Simplificaciones vs AlphaZero original:

1. **Encoding de movimientos:** Simplificado (4672 slots vs ~4672 con direcciones)
2. **Planos de estado:** Solo posición actual, sin historial de 8 jugadas
3. **Tamaño de red:** 10 bloques vs 20-40 del original
4. **Auto-juego:** 100 partidas/iter vs 25,000 del original
5. **Hardware:** CPU/GPU simple vs TPU farms
6. **Evaluación:** Sin "arena" para seleccionar mejor versión

**Pero es suficiente para:** Demostrar el concepto, aprender funcionamiento, jugar nivel intermedio

---

### P10: ¿Cómo sabes que está aprendiendo?

**R:** Métricas observables:

1. **Pérdida:** Debe disminuir con iteraciones
   ```python
   logger.info("Pérdida promedio: %.4f", np.mean(perdidas))
   ```

2. **Profundidad de juego:** Partidas más largas = menos errores graves

3. **Evaluación:** Hacer que juegue contra versión anterior
   - Win rate > 55% → mejora significativa
   - Win rate < 45% → regresión

4. **Calidad de política:** Probabilidades altas en jugadas reconocidamente buenas

5. **Estabilidad de valor:** Predicciones consistentes con resultados

**Código de monitoreo:** [`monitor_entrenamiento.py`](monitor_entrenamiento.py)

---

## 7. Cómo Ejecutar y Demostrar

### 7.1 Entrenamiento Rápido (Demo)
```powershell
# Configuración mínima para demostración (5-10 min)
python entrenar_rapido_alphazero.py
```

Parámetros reducidos:
- 5 iteraciones
- 10 partidas por iteración
- 100 simulaciones por jugada

### 7.2 Ver AlphaZero Jugar
```powershell
# Ver una partida con el modelo entrenado
python ver_alphazero_jugar.py
```

Muestra:
- Tablero ASCII por jugada
- Valor estimado de la posición
- Tiempo de búsqueda
- Jugada seleccionada

### 7.3 Comparar vs Otros Agentes
```powershell
# AlphaZero vs Minimax
python ejecutar_minimax_vs_mcts.py
```

### 7.4 Verificar Estado de Entrenamiento
```powershell
# Monitorear pérdidas y métricas
python monitor_entrenamiento.py
```

---

## 8. Puntos Clave para la Presentación

### Lo Más Importante (memoriza esto):

1. **AlphaZero = MCTS + Red Neuronal + Auto-juego**
   - MCTS busca las mejores jugadas
   - Red neuronal guía y acelera esa búsqueda
   - Auto-juego genera datos para mejorar la red

2. **La red predice dos cosas:**
   - Política (qué jugadas son buenas)
   - Valor (quién va ganando)

3. **MCTS usa esas predicciones:**
   - Política → Explora jugadas prometedoras (prior P)
   - Valor → Evalúa hojas del árbol rápidamente

4. **Entrenamiento por imitación:**
   - Política aprende de las visitas MCTS
   - Valor aprende del resultado final

5. **Mejora iterativa:**
   - Mejor red → Mejor MCTS → Mejores datos → Mejor red

### Frase Final:
> "AlphaZero demuestra que es posible aprender estrategia de nivel maestro sin conocimiento humano, solo mediante auto-juego guiado por búsqueda y redes neuronales."

---

## 9. Recursos Adicionales

### Archivos clave del proyecto:
- [`agentes/agente_alphazero.py`](agentes/agente_alphazero.py) - Implementación completa del agente
- [`entrenamiento/entrenar_alphazero.py`](entrenamiento/entrenar_alphazero.py) - Pipeline de entrenamiento
- [`entorno/entorno_ajedrez.py`](entorno/entorno_ajedrez.py) - Interfaz del juego
- [`ALPHAZERO_INFO.md`](ALPHAZERO_INFO.md) - Documentación técnica adicional

### Papers originales:
1. Silver et al. (2017) - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
2. Silver et al. (2018) - "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"

---

## 10. Checklist Pre-Presentación

- [ ] Entender flujo completo: tablero → tensor → red → política/valor → MCTS → jugada
- [ ] Explicar fórmula PUCT con Q + U
- [ ] Diferenciar rol de política vs valor
- [ ] Describir ciclo de entrenamiento (auto-juego → buffer → batches)
- [ ] Justificar cambio de signo en retropropagación
- [ ] Conocer limitaciones de esta implementación vs original
- [ ] Tener modelo entrenado listo para demo
- [ ] Preparar respuestas a "¿por qué no usar solo MCTS?" y "¿por qué no solo la red?"

---

**¡Buena suerte en tu presentación!** 🎯♟️
