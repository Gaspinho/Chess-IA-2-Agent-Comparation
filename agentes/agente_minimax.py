"""
Agente de ajedrez basado en Minimax con poda alfa-beta.
"""

import chess
import chess.engine
import numpy as np
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AgenteMinimax:
    """
    Agente de ajedrez que usa el algoritmo Minimax con poda alfa-beta.
    """
    
    def __init__(self, profundidad_maxima: int = 3, tiempo_limite: float = 5.0):
        """
        Inicializa el agente Minimax.
        
        Args:
            profundidad_maxima: Profundidad máxima de búsqueda
            tiempo_limite: Tiempo límite por jugada en segundos
        """
        self.profundidad_maxima = profundidad_maxima
        self.tiempo_limite = tiempo_limite
        self.nombre = f"Minimax_Prof{profundidad_maxima}"
        self.nodos_evaluados = 0
        self.tiempo_inicio = 0
        
        # Valores de las piezas para evaluación
        self.valores_piezas = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Tablas de posición para bonificaciones posicionales
        self.tabla_peones = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.tabla_caballos = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
    
    def evaluar_tablero(self, tablero: chess.Board) -> float:
        """
        Evalúa la posición del tablero desde la perspectiva de las blancas.
        
        Args:
            tablero: Estado del tablero a evaluar
            
        Returns:
            float: Puntuación de la posición (positiva para blancas)
        """
        if tablero.is_checkmate():
            return -999999 if tablero.turn else 999999
        
        if tablero.is_stalemate() or tablero.is_insufficient_material():
            return 0
        
        puntuacion = 0
        
        # Evaluar material y posición
        for cuadrado in chess.SQUARES:
            pieza = tablero.piece_at(cuadrado)
            if pieza is None:
                continue
                
            valor = self.valores_piezas[pieza.piece_type]
            
            # Bonificaciones posicionales
            if pieza.piece_type == chess.PAWN:
                if pieza.color == chess.WHITE:
                    valor += self.tabla_peones[cuadrado]
                else:
                    valor += self.tabla_peones[chess.square_mirror(cuadrado)]
            elif pieza.piece_type == chess.KNIGHT:
                if pieza.color == chess.WHITE:
                    valor += self.tabla_caballos[cuadrado]
                else:
                    valor += self.tabla_caballos[chess.square_mirror(cuadrado)]
            
            # Aplicar signo según color
            if pieza.color == chess.WHITE:
                puntuacion += valor
            else:
                puntuacion -= valor
        
        # Bonificaciones adicionales
        puntuacion += self._evaluar_movilidad(tablero)
        puntuacion += self._evaluar_seguridad_rey(tablero)
        puntuacion += self._evaluar_estructura_peones(tablero)
        
        return puntuacion
    
    def _evaluar_movilidad(self, tablero: chess.Board) -> float:
        """Evalúa la movilidad de las piezas."""
        movimientos_legales = len(list(tablero.legal_moves))
        
        # Cambiar turno para contar movimientos del oponente
        tablero.push(chess.Move.null())
        if tablero.is_valid():
            movimientos_oponente = len(list(tablero.legal_moves))
            tablero.pop()
        else:
            tablero.pop()
            movimientos_oponente = 0
        
        movilidad = movimientos_legales - movimientos_oponente
        return movilidad * 0.1 if tablero.turn == chess.WHITE else -movilidad * 0.1
    
    def _evaluar_seguridad_rey(self, tablero: chess.Board) -> float:
        """Evalúa la seguridad del rey."""
        puntuacion = 0
        
        # Rey blanco
        rey_blanco = tablero.king(chess.WHITE)
        if rey_blanco:
            if tablero.is_attacked_by(chess.BLACK, rey_blanco):
                puntuacion -= 50
        
        # Rey negro
        rey_negro = tablero.king(chess.BLACK)
        if rey_negro:
            if tablero.is_attacked_by(chess.WHITE, rey_negro):
                puntuacion += 50
        
        return puntuacion
    
    def _evaluar_estructura_peones(self, tablero: chess.Board) -> float:
        """Evalúa la estructura de peones."""
        puntuacion = 0
        
        # Peones doblados, aislados, etc.
        for archivo in range(8):
            peones_blancos = 0
            peones_negros = 0
            
            for fila in range(8):
                cuadrado = chess.square(archivo, fila)
                pieza = tablero.piece_at(cuadrado)
                
                if pieza and pieza.piece_type == chess.PAWN:
                    if pieza.color == chess.WHITE:
                        peones_blancos += 1
                    else:
                        peones_negros += 1
            
            # Penalizar peones doblados
            if peones_blancos > 1:
                puntuacion -= 10 * (peones_blancos - 1)
            if peones_negros > 1:
                puntuacion += 10 * (peones_negros - 1)
        
        return puntuacion
    
    def minimax(self, tablero: chess.Board, profundidad: int, alfa: float, beta: float, maximizando: bool) -> Tuple[float, Optional[chess.Move]]:
        """
        Implementa el algoritmo Minimax con poda alfa-beta.
        
        Args:
            tablero: Estado actual del tablero
            profundidad: Profundidad restante de búsqueda
            alfa: Valor alfa para poda
            beta: Valor beta para poda
            maximizando: True si está maximizando, False si minimizando
            
        Returns:
            tuple: (mejor_puntuacion, mejor_movimiento)
        """
        self.nodos_evaluados += 1
        
        # Verificar límite de tiempo
        if time.time() - self.tiempo_inicio > self.tiempo_limite:
            return self.evaluar_tablero(tablero), None
        
        # Condiciones de parada
        if profundidad == 0 or tablero.is_game_over():
            return self.evaluar_tablero(tablero), None
        
        movimientos = list(tablero.legal_moves)
        if not movimientos:
            return self.evaluar_tablero(tablero), None
        
        # Ordenar movimientos para mejorar poda (capturas primero)
        movimientos = self._ordenar_movimientos(tablero, movimientos)
        
        mejor_movimiento = None
        
        if maximizando:
            max_eval = float('-inf')
            for movimiento in movimientos:
                tablero.push(movimiento)
                eval_actual, _ = self.minimax(tablero, profundidad - 1, alfa, beta, False)
                tablero.pop()
                
                if eval_actual > max_eval:
                    max_eval = eval_actual
                    mejor_movimiento = movimiento
                
                alfa = max(alfa, eval_actual)
                if beta <= alfa:
                    break  # Poda alfa-beta
            
            return max_eval, mejor_movimiento
        else:
            min_eval = float('inf')
            for movimiento in movimientos:
                tablero.push(movimiento)
                eval_actual, _ = self.minimax(tablero, profundidad - 1, alfa, beta, True)
                tablero.pop()
                
                if eval_actual < min_eval:
                    min_eval = eval_actual
                    mejor_movimiento = movimiento
                
                beta = min(beta, eval_actual)
                if beta <= alfa:
                    break  # Poda alfa-beta
            
            return min_eval, mejor_movimiento
    
    def _ordenar_movimientos(self, tablero: chess.Board, movimientos: list) -> list:
        """
        Ordena los movimientos para mejorar la eficiencia de la poda alfa-beta.
        
        Args:
            tablero: Estado actual del tablero
            movimientos: Lista de movimientos legales
            
        Returns:
            list: Movimientos ordenados por prioridad
        """
        def prioridad_movimiento(movimiento):
            puntuacion = 0
            
            # Capturas tienen alta prioridad
            if tablero.is_capture(movimiento):
                pieza_capturada = tablero.piece_at(movimiento.to_square)
                if pieza_capturada:
                    puntuacion += self.valores_piezas[pieza_capturada.piece_type]
            
            # Jaques tienen prioridad
            tablero.push(movimiento)
            if tablero.is_check():
                puntuacion += 50
            tablero.pop()
            
            # Promociones tienen alta prioridad
            if movimiento.promotion:
                puntuacion += 800
            
            return puntuacion
        
        return sorted(movimientos, key=prioridad_movimiento, reverse=True)
    
    def seleccionar_movimiento(self, tablero: chess.Board) -> Optional[chess.Move]:
        """
        Selecciona el mejor movimiento usando Minimax.
        
        Args:
            tablero: Estado actual del tablero
            
        Returns:
            chess.Move: Mejor movimiento encontrado
        """
        self.tiempo_inicio = time.time()
        self.nodos_evaluados = 0
        
        logger.info(f"Agente {self.nombre} calculando movimiento...")
        
        try:
            # Búsqueda iterativa por profundidad
            mejor_movimiento = None
            
            for profundidad in range(1, self.profundidad_maxima + 1):
                if time.time() - self.tiempo_inicio > self.tiempo_limite * 0.8:
                    break
                
                _, movimiento = self.minimax(
                    tablero, 
                    profundidad, 
                    float('-inf'), 
                    float('inf'), 
                    tablero.turn == chess.WHITE
                )
                
                if movimiento:
                    mejor_movimiento = movimiento
                
                logger.debug(f"Profundidad {profundidad} completada")
            
            tiempo_total = time.time() - self.tiempo_inicio
            logger.info(f"Movimiento calculado en {tiempo_total:.2f}s, "
                       f"evaluando {self.nodos_evaluados} nodos")
            
            return mejor_movimiento
            
        except Exception as e:
            logger.error(f"Error en selección de movimiento: {e}")
            # Fallback: movimiento aleatorio
            movimientos = list(tablero.legal_moves)
            return movimientos[0] if movimientos else None
    
    def obtener_estadisticas(self) -> dict:
        """
        Obtiene estadísticas del último cálculo.
        
        Returns:
            dict: Estadísticas del agente
        """
        return {
            'nombre': self.nombre,
            'profundidad_maxima': self.profundidad_maxima,
            'nodos_evaluados': self.nodos_evaluados,
            'tiempo_ultimo_movimiento': time.time() - self.tiempo_inicio if self.tiempo_inicio else 0
        }


def convertir_accion_a_movimiento(accion: int, tablero: chess.Board) -> Optional[chess.Move]:
    """
    Convierte una acción numérica en un movimiento de chess.
    
    Args:
        accion: Índice de la acción
        tablero: Estado actual del tablero
        
    Returns:
        chess.Move: Movimiento correspondiente o None
    """
    try:
        movimientos = list(tablero.legal_moves)
        if 0 <= accion < len(movimientos):
            return movimientos[accion]
        return None
    except Exception:
        return None


def convertir_movimiento_a_accion(movimiento: chess.Move, tablero: chess.Board) -> Optional[int]:
    """
    Convierte un movimiento de chess en una acción numérica.
    
    Args:
        movimiento: Movimiento de chess
        tablero: Estado actual del tablero
        
    Returns:
        int: Índice de la acción o None
    """
    try:
        movimientos = list(tablero.legal_moves)
        return movimientos.index(movimiento)
    except (ValueError, Exception):
        return None


if __name__ == "__main__":
    # Prueba del agente Minimax
    print("Probando agente Minimax...")
    
    tablero = chess.Board()
    agente = AgenteMinimax(profundidad_maxima=3)
    
    print("Posición inicial:")
    print(tablero)
    
    movimiento = agente.seleccionar_movimiento(tablero)
    print(f"Mejor movimiento: {movimiento}")
    
    stats = agente.obtener_estadisticas()
    print(f"Estadísticas: {stats}")
    
    print("Prueba completada")
