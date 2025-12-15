"""
Agente Negamax Avanzado para ajedrez.
Implementa Negamax con poda alfa-beta, tabla de transposición,
ordenación de movimientos, quiescence search y control de tiempo.
"""

import logging
import time
import chess
import chess.polyglot

# Configuración de logging para monitorear el rendimiento del agente
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes para las banderas de la Tabla de Transposición (TT)
# Ayudan a distinguir si el valor guardado es exacto o una cota
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

# Límite de entradas en la tabla de transposición para controlar memoria
MAX_TT_SIZE = 2_000_000

# Valores de puntuación para situaciones de Jaque Mate
MATE_SCORE = 20000
MATE_THRESHOLD = 19000


class AgenteNegamaxAvanzado:

    def __init__(self, tiempo_limite: float = 5.0):
        """
        Inicializa el agente con un tiempo límite por jugada.
        Configura las estructuras de datos para estadísticas y evaluación.
        """
        self.tiempo_limite = tiempo_limite
        self.nombre = "Negamax Avanzado"
        self.tt = {}  # Tabla de transposición

        # Estadísticas para evaluar el desempeño de la búsqueda
        self.stats = {
            'nodos': 0,
            'tt_hits': 0,
            'profundidad_alcanzada': 0,
            'tiempo_total': 0.0,
            'q_nodos': 0
        }

        # Valores materiales estándar de las piezas (centipeones)
        self.valores_piezas = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Piece-Square Tables (PST): Tablas de valoración posicional.
        # Definen bonificaciones/penalizaciones según la casilla que ocupa la pieza.
        # Definidas desde la perspectiva de las blancas.
        
        # Peones: Incentiva avanzar y controlar el centro.
        self.pst_peon = list(reversed([
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ]))

        # Caballos: Se penalizan en los bordes, se premian en el centro.
        self.pst_caballo = list(reversed([
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]))

        # Alfiles: Se valoran mejor en diagonales largas y control central.
        self.pst_alfil = list(reversed([
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]))

        # Torres: Preferencia por la séptima fila y columnas centrales.
        self.pst_torre = list(reversed([
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  0,  5,  5,  0,  0,  0
        ]))

        # Reina: Movilidad central, evitando exposición temprana en las esquinas.
        self.pst_reina = list(reversed([
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]))

        # Rey: Protección en el enroque (esquinas inferiores) en juego medio.
        self.pst_rey = list(reversed([
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ]))

    # =========================
    # Selección de movimiento
    # =========================

    def seleccionar_accion(self, observacion, acciones_legales, info):
        """
        Método principal llamado para obtener el siguiente movimiento.
        Implementa 'Iterative Deepening' (Profundización Iterativa):
        Busca a profundidad 1, luego 2, etc., hasta que se agote el tiempo.
        """
        inicio = time.time()
        self._reset_stats()

        # Limpieza de la tabla de transposición si crece demasiado
        if len(self.tt) > MAX_TT_SIZE:
            self.tt.clear()

        # Reconstrucción del tablero a partir de la info
        tablero = info.get('board', chess.Board(info.get('fen', chess.STARTING_FEN)))

        mejor_movimiento = None
        profundidad = 1

        # Determina el multiplicador para Negamax (1 para blancas, -1 para negras)
        color_raiz = 1 if tablero.turn == chess.WHITE else -1

        try:
            while True:
                # Verificación de seguridad antes de iniciar una nueva profundidad
                if time.time() - inicio > self.tiempo_limite:
                    break

                # Ejecuta la búsqueda raíz para la profundidad actual
                valor, mov = self._busqueda_raiz(
                    tablero, profundidad, inicio, color_raiz
                )

                mejor_movimiento = mov
                self.stats['profundidad_alcanzada'] = profundidad

                # Si encontramos un mate forzado, detenemos la búsqueda
                if abs(valor) > MATE_THRESHOLD:
                    break

                profundidad += 1
                # Límite de seguridad para evitar loops infinitos en endgames simples
                if profundidad > 100:
                    break

        except TimeoutError:
            # Si se lanza la excepción, retornamos el mejor movimiento de la iteración completa anterior
            pass

        # Fallback: si no se encontró nada (caso raro), toma el primer legal
        if mejor_movimiento is None:
            movs = list(tablero.legal_moves)
            if movs:
                mejor_movimiento = movs[0]

        self.stats['tiempo_total'] = time.time() - inicio

        logger.info(
            f"NegamaxAv (D{self.stats['profundidad_alcanzada']}): "
            f"{mejor_movimiento} | "
            f"Nodos: {self.stats['nodos']} | "
            f"TT: {self.stats['tt_hits']}"
        )

        return mejor_movimiento

    # =========================
    # Búsqueda Negamax
    # =========================

    def _busqueda_raiz(self, tablero, profundidad, inicio_tiempo, color):
        """
        Primera capa de la búsqueda. Separa la lógica de la raíz para manejar
        el retorno del movimiento (y no solo el puntaje).
        """
        alpha = -float('inf')
        beta = float('inf')
        mejor_mov = None

        # Ordenamos movimientos para intentar primero los más prometedores
        movimientos = self._ordenar_movimientos(
            tablero, list(tablero.legal_moves), None
        )

        for mov in movimientos:
            tablero.push(mov)
            # Llamada recursiva a negamax invirtiendo alpha/beta y color
            valor = -self._negamax(
                tablero, profundidad - 1, -beta, -alpha,
                -color, inicio_tiempo, 1
            )
            tablero.pop()

            if valor > alpha:
                alpha = valor
                mejor_mov = mov

        return alpha, mejor_mov

    def _negamax(self, tablero, profundidad, alpha, beta, color,
                 inicio_tiempo, ply):
        """
        Algoritmo Negamax con poda Alfa-Beta y Tabla de Transposición.
        """

        # Verificación de tiempo: solo cada 2048 nodos para optimizar rendimiento
        if self.stats['nodos'] & 2047 == 0:
            if time.time() - inicio_tiempo > self.tiempo_limite:
                raise TimeoutError()

        self.stats['nodos'] += 1
        alpha_original = alpha

        # --- Tabla de transposición ---
        # Calculamos el hash Zobrist de la posición actual
        key = chess.polyglot.zobrist_hash(tablero)
        entry = self.tt.get(key)

        if entry:
            tt_depth, tt_value, tt_flag, tt_move = entry

            # Si la entrada en la tabla es de una búsqueda igual o más profunda, es útil
            if tt_depth >= profundidad:
                score = tt_value
                # Ajuste de puntaje de mate según la distancia actual (ply)
                if abs(score) > MATE_THRESHOLD:
                    score += -ply if score > 0 else ply

                if tt_flag == TT_EXACT:
                    self.stats['tt_hits'] += 1
                    return score
                elif tt_flag == TT_LOWERBOUND:
                    alpha = max(alpha, score)
                elif tt_flag == TT_UPPERBOUND:
                    beta = min(beta, score)

                # Si logramos una poda gracias a la tabla, retornamos
                if alpha >= beta:
                    self.stats['tt_hits'] += 1
                    return alpha

        # --- Estado Terminal ---
        if tablero.is_game_over():
            return self._evaluar_terminal(tablero, color, ply)

        # --- Quiescence Search ---
        # Si llegamos a profundidad 0, seguimos buscando solo capturas para estabilidad
        if profundidad <= 0:
            return self._quiescence(tablero, alpha, beta, color)

        # --- Recursión ---
        mejor_valor = -float('inf')
        mejor_mov = None

        # Usamos el movimiento de la tabla de transposición (si existe) para ordenar mejor
        movimientos = self._ordenar_movimientos(
            tablero, list(tablero.legal_moves),
            entry[3] if entry else None
        )

        for mov in movimientos:
            tablero.push(mov)
            valor = -self._negamax(
                tablero, profundidad - 1,
                -beta, -alpha, -color,
                inicio_tiempo, ply + 1
            )
            tablero.pop()

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_mov = mov

            # Actualización de la cota inferior
            alpha = max(alpha, valor)
            # Poda Beta
            if alpha >= beta:
                break

        # --- Guardado en Tabla de Transposición ---
        # Determinamos el tipo de bandera según cómo terminó la búsqueda
        if mejor_valor <= alpha_original:
            flag = TT_UPPERBOUND
        elif mejor_valor >= beta:
            flag = TT_LOWERBOUND
        else:
            flag = TT_EXACT

        # Ajuste inverso del mate para almacenarlo independiente del ply
        save_val = mejor_valor
        if abs(save_val) > MATE_THRESHOLD:
            save_val += ply if save_val > 0 else -ply

        self.tt[key] = (profundidad, save_val, flag, mejor_mov)

        return mejor_valor

    # =========================
    # Quiescence Search
    # =========================

    def _quiescence(self, tablero, alpha, beta, color):
        """
        Búsqueda de quietud para mitigar el 'efecto horizonte'.
        Solo explora capturas hasta encontrar una posición estable.
        """
        self.stats['q_nodos'] += 1

        # Evaluación estática actual (Stand Pat)
        # Asumimos que el jugador puede decidir no capturar si no le conviene
        stand_pat = color * self._evaluar_posicion(tablero)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generación y ordenamiento exclusivo de capturas y promociones
        capturas = [
            m for m in tablero.legal_moves
            if tablero.is_capture(m) or m.promotion
        ]

        capturas = self._ordenar_movimientos(
            tablero, capturas, None, solo_capturas=True
        )

        for mov in capturas:
            tablero.push(mov)
            score = -self._quiescence(tablero, -beta, -alpha, -color)
            tablero.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # =========================
    # Evaluación
    # =========================

    def _evaluar_posicion(self, tablero):
        """
        Función de evaluación estática.
        Combina valor material y tablas de posición (PST).
        """
        if tablero.is_insufficient_material():
            return 0

        score = 0
        psts = {
            chess.PAWN: self.pst_peon,
            chess.KNIGHT: self.pst_caballo,
            chess.BISHOP: self.pst_alfil,
            chess.ROOK: self.pst_torre,
            chess.QUEEN: self.pst_reina,
            chess.KING: self.pst_rey
        }

        # Iteramos sobre todas las piezas en el tablero
        for sq, pieza in tablero.piece_map().items():
            valor = self.valores_piezas[pieza.piece_type]
            tabla = psts[pieza.piece_type]

            if pieza.color == chess.WHITE:
                # Mapeo directo para blancas
                r = chess.square_rank(sq)
                f = chess.square_file(sq)
                idx = (7 - r) * 8 + f
                score += valor + tabla[idx]
            else:
                # Para negras, usamos espejo (mirror) para leer la tabla correctamente
                sq_m = chess.square_mirror(sq)
                r = chess.square_rank(sq_m)
                f = chess.square_file(sq_m)
                idx = (7 - r) * 8 + f
                score -= valor + tabla[idx]

        return score

    def _evaluar_terminal(self, tablero, color, ply):
        """
        Retorna puntaje si el juego ha terminado (Mate o Tablas).
        """
        if tablero.is_checkmate():
            # Preferimos mates más rápidos (menos ply)
            return -color * (MATE_SCORE - ply)

        return 0

    # =========================
    # Utilidades
    # =========================

    def _ordenar_movimientos(self, tablero, movimientos, tt_move,
                             solo_capturas=False):
        """
        Heurística de ordenamiento de movimientos para mejorar la poda Alpha-Beta.
        Orden:
        1. Movimiento de la Tabla de Transposición (Hash Move).
        2. Promociones.
        3. Capturas (MVV-LVA: Víctima valiosa, Atacante de poco valor).
        """

        def score_move(m):
            if m == tt_move:
                return 10_000_000  # Prioridad máxima

            score = 0

            if m.promotion:
                score += 900_000

            if tablero.is_capture(m):
                # MVV-LVA logic
                if tablero.is_en_passant(m):
                    val_victima = 100
                else:
                    tipo = tablero.piece_type_at(m.to_square)
                    val_victima = self.valores_piezas.get(tipo, 0)

                atacante = tablero.piece_type_at(m.from_square)
                val_atacante = self.valores_piezas.get(atacante, 0)

                # Fórmula para priorizar capturas ventajosas
                score += 100_000 + (val_victima * 10 - val_atacante)

            return score

        movimientos.sort(key=score_move, reverse=True)
        return movimientos

    def _movimiento_a_accion(self, movimiento, tablero, acciones_legales):
        """
        Mapea el movimiento de chess.py al formato de acciones requerido por el entorno.
        """
        if not acciones_legales:
            return 0

        if movimiento is None:
            return acciones_legales[0]

        movs = list(tablero.legal_moves)
        if movimiento in movs:
            idx = movs.index(movimiento)
            if idx < len(acciones_legales):
                return acciones_legales[idx]

        return acciones_legales[0]

    def _reset_stats(self):
        """Reinicia los contadores para la nueva búsqueda."""
        self.stats = {
            'nodos': 0,
            'tt_hits': 0,
            'profundidad_alcanzada': 0,
            'tiempo_total': 0.0,
            'q_nodos': 0
        }