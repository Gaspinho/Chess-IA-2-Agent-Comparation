"""
Agente Minimax con poda alfa-beta para ajedrez.

Este módulo implementa un agente que utiliza el algoritmo Minimax con 
poda alfa-beta para evaluar posiciones y seleccionar las mejores jugadas.
"""

import logging
import time
import numpy as np
import chess
import chess.engine
from typing import Optional, Tuple, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenteMinimax:
    """
    Agente que utiliza Minimax con poda alfa-beta para jugar ajedrez.
    
    Este agente evalúa posiciones utilizando una función de evaluación
    heurística y busca la mejor jugada usando el algoritmo Minimax.
    """
    
    def __init__(self, profundidad_maxima: int = 3, tiempo_limite: float = 5.0):
        """
        Inicializa el agente Minimax.
        
        Args:
            profundidad_maxima: Profundidad máxima de búsqueda
            tiempo_limite: Tiempo límite en segundos para cada jugada
        """
        self.profundidad_maxima = profundidad_maxima
        self.tiempo_limite = tiempo_limite
        self.nombre = "Minimax"
        self.estadisticas = {
            'nodos_evaluados': 0,
            'podas_alfa': 0,
            'podas_beta': 0,
            'tiempo_total': 0.0,
            'jugadas_realizadas': 0
        }
        
        # Valores de las piezas
        self.valores_piezas = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Tablas de posición para piezas (simplified)
        self.tabla_peones = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ])
        
        self.tabla_caballos = np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ])
        
        logger.info("Agente Minimax inicializado - Profundidad: %d, Tiempo límite: %.1fs", 
                   profundidad_maxima, tiempo_limite)
    
    def seleccionar_accion(self, observacion: np.ndarray, acciones_legales: np.ndarray, 
                          info: Dict[str, Any]) -> int:
        """
        Selecciona la mejor acción usando Minimax con poda alfa-beta.
        
        Args:
            observacion: Estado actual del juego
            acciones_legales: Acciones legales disponibles
            info: Información adicional del entorno
            
        Returns:
            Índice de la acción seleccionada
        """
        inicio_tiempo = time.time()
        
        try:
            # Resetear estadísticas para esta jugada
            self.estadisticas['nodos_evaluados'] = 0
            self.estadisticas['podas_alfa'] = 0
            self.estadisticas['podas_beta'] = 0
            
            if len(acciones_legales) == 0:
                logger.warning("No hay acciones legales disponibles")
                return 0
            
            if len(acciones_legales) == 1:
                logger.info("Solo una acción legal disponible")
                return acciones_legales[0]
            
            # Obtener tablero actual desde info si está disponible
            tablero = self._obtener_tablero_desde_info(info)
            if tablero is None:
                # Fallback: selección aleatoria
                logger.warning("No se pudo obtener estado del tablero, selección aleatoria")
                return np.random.choice(acciones_legales)
            
            mejor_accion = None
            mejor_valor = float('-inf')
            
            # Evaluar cada acción legal
            for accion in acciones_legales:
                # Convertir acción a movimiento si es posible
                movimiento = self._accion_a_movimiento(accion, tablero)
                if movimiento is None:
                    continue
                
                # Hacer movimiento temporal
                tablero.push(movimiento)
                
                # Evaluar posición resultante
                valor = self._minimax(tablero, self.profundidad_maxima - 1, 
                                    float('-inf'), float('inf'), False, inicio_tiempo)
                
                # Deshacer movimiento
                tablero.pop()
                
                if valor > mejor_valor:
                    mejor_valor = valor
                    mejor_accion = accion
                
                # Verificar tiempo límite
                if time.time() - inicio_tiempo > self.tiempo_limite:
                    logger.warning("Tiempo límite alcanzado en selección de acción")
                    break
            
            tiempo_jugada = time.time() - inicio_tiempo
            self.estadisticas['tiempo_total'] += tiempo_jugada
            self.estadisticas['jugadas_realizadas'] += 1
            
            logger.debug("Minimax - Mejor acción: %s, Valor: %.2f, Nodos: %d, Tiempo: %.3fs",
                        mejor_accion, mejor_valor, self.estadisticas['nodos_evaluados'], tiempo_jugada)
            
            return mejor_accion if mejor_accion is not None else acciones_legales[0]
            
        except Exception as e:
            logger.error("Error en selección de acción Minimax: %s", str(e))
            return np.random.choice(acciones_legales)
    
    def _minimax(self, tablero: chess.Board, profundidad: int, alfa: float, beta: float, 
                maximizar: bool, inicio_tiempo: float) -> float:
        """
        Implementación del algoritmo Minimax con poda alfa-beta.
        
        Args:
            tablero: Estado actual del tablero
            profundidad: Profundidad restante de búsqueda
            alfa: Valor alfa para poda
            beta: Valor beta para poda
            maximizar: True si es turno del maximizador
            inicio_tiempo: Tiempo de inicio para control de tiempo
            
        Returns:
            Valor de evaluación de la posición
        """
        self.estadisticas['nodos_evaluados'] += 1
        
        # Verificar tiempo límite
        if time.time() - inicio_tiempo > self.tiempo_limite:
            return self._evaluar_posicion(tablero)
        
        # Condiciones de terminación
        if profundidad == 0 or tablero.is_game_over():
            return self._evaluar_posicion(tablero)
        
        if maximizar:
            max_eval = float('-inf')
            for movimiento in tablero.legal_moves:
                tablero.push(movimiento)
                eval_val = self._minimax(tablero, profundidad - 1, alfa, beta, False, inicio_tiempo)
                tablero.pop()
                
                max_eval = max(max_eval, eval_val)
                alfa = max(alfa, eval_val)
                
                if beta <= alfa:
                    self.estadisticas['podas_beta'] += 1
                    break  # Poda beta
                    
            return max_eval
        else:
            min_eval = float('inf')
            for movimiento in tablero.legal_moves:
                tablero.push(movimiento)
                eval_val = self._minimax(tablero, profundidad - 1, alfa, beta, True, inicio_tiempo)
                tablero.pop()
                
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                
                if beta <= alfa:
                    self.estadisticas['podas_alfa'] += 1
                    break  # Poda alfa
                    
            return min_eval
    
    def _evaluar_posicion(self, tablero: chess.Board) -> float:
        """
        Evalúa la posición actual del tablero.
        
        Args:
            tablero: Estado del tablero a evaluar
            
        Returns:
            Valor de evaluación (positivo favorable a blancas)
        """
        if tablero.is_checkmate():
            return -20000 if tablero.turn else 20000
        
        if tablero.is_stalemate() or tablero.is_insufficient_material():
            return 0
        
        puntuacion = 0
        
        # Evaluar material y posición
        for cuadro in chess.SQUARES:
            pieza = tablero.piece_at(cuadro)
            if pieza is not None:
                valor = self.valores_piezas[pieza.piece_type]
                
                # Añadir valor posicional
                valor += self._obtener_valor_posicional(pieza, cuadro)
                
                if pieza.color == chess.WHITE:
                    puntuacion += valor
                else:
                    puntuacion -= valor
        
        # Bonificaciones adicionales
        puntuacion += self._evaluar_estructura_peones(tablero)
        puntuacion += self._evaluar_seguridad_rey(tablero)
        puntuacion += self._evaluar_movilidad(tablero)
        
        return puntuacion
    
    def _obtener_valor_posicional(self, pieza: chess.Piece, cuadro: int) -> float:
        """
        Obtiene el valor posicional de una pieza en un cuadro específico.
        
        Args:
            pieza: Pieza a evaluar
            cuadro: Cuadro donde está la pieza
            
        Returns:
            Valor posicional
        """
        fila = chess.square_rank(cuadro)
        columna = chess.square_file(cuadro)
        
        if not pieza.color:  # Negras
            fila = 7 - fila
        
        if pieza.piece_type == chess.PAWN:
            return self.tabla_peones[fila][columna]
        elif pieza.piece_type == chess.KNIGHT:
            return self.tabla_caballos[fila][columna]
        
        return 0
    
    def _evaluar_estructura_peones(self, tablero: chess.Board) -> float:
        """Evalúa la estructura de peones."""
        puntuacion = 0
        
        # Peones doblados, aislados, etc.
        for columna in range(8):
            peones_blancos = 0
            peones_negros = 0
            
            for fila in range(8):
                cuadro = chess.square(columna, fila)
                pieza = tablero.piece_at(cuadro)
                
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
    
    def _evaluar_seguridad_rey(self, tablero: chess.Board) -> float:
        """Evalúa la seguridad del rey."""
        puntuacion = 0
        
        # Bonificar enroque
        if tablero.has_castling_rights(chess.WHITE):
            puntuacion += 30
        if tablero.has_castling_rights(chess.BLACK):
            puntuacion -= 30
        
        return puntuacion
    
    def _evaluar_movilidad(self, tablero: chess.Board) -> float:
        """Evalúa la movilidad de las piezas."""
        movilidad_blancas = len(list(tablero.legal_moves))
        
        # Cambiar turno temporalmente
        tablero.push(chess.Move.null())
        movilidad_negras = len(list(tablero.legal_moves))
        tablero.pop()
        
        return (movilidad_blancas - movilidad_negras) * 0.1
    
    def _obtener_tablero_desde_info(self, info: Dict[str, Any]) -> Optional[chess.Board]:
        """
        Intenta obtener el estado del tablero desde la información del entorno.
        
        Args:
            info: Información del entorno
            
        Returns:
            Objeto Board de chess o None si no se puede obtener
        """
        try:
            # Intentar diferentes formas de obtener el tablero
            if 'board' in info:
                return info['board']
            
            if 'fen' in info:
                return chess.Board(info['fen'])
            
            # Si no hay información específica, crear tablero inicial
            return chess.Board()
            
        except Exception as e:
            logger.error("Error al obtener tablero desde info: %s", str(e))
            return None
    
    def _accion_a_movimiento(self, accion: int, tablero: chess.Board) -> Optional[chess.Move]:
        """
        Convierte una acción numérica a un movimiento de chess.
        
        Args:
            accion: Índice de acción
            tablero: Estado del tablero
            
        Returns:
            Movimiento correspondiente o None si no es válido
        """
        try:
            movimientos_legales = list(tablero.legal_moves)
            if 0 <= accion < len(movimientos_legales):
                return movimientos_legales[accion]
            return None
            
        except Exception as e:
            logger.error("Error al convertir acción a movimiento: %s", str(e))
            return None
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene las estadísticas del agente.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.estadisticas.copy()
        if stats['jugadas_realizadas'] > 0:
            stats['tiempo_promedio_por_jugada'] = stats['tiempo_total'] / stats['jugadas_realizadas']
            stats['nodos_promedio_por_jugada'] = stats['nodos_evaluados'] / stats['jugadas_realizadas']
        else:
            stats['tiempo_promedio_por_jugada'] = 0
            stats['nodos_promedio_por_jugada'] = 0
        
        return stats
    
    def obtener_mejor_jugada(self, tablero: chess.Board) -> chess.Move:
        """
        Obtiene la mejor jugada para un tablero de chess directamente.
        
        Args:
            tablero: Tablero de chess
            
        Returns:
            Mejor movimiento encontrado
        """
        inicio_tiempo = time.time()
        
        try:
            # Resetear estadísticas para esta jugada
            self.estadisticas['nodos_evaluados'] = 0
            self.estadisticas['podas_alfa'] = 0
            self.estadisticas['podas_beta'] = 0
            
            movimientos_legales = list(tablero.legal_moves)
            
            if len(movimientos_legales) == 1:
                # Solo un movimiento legal
                return movimientos_legales[0]
            
            mejor_movimiento = None
            mejor_valor = float('-inf') if tablero.turn else float('inf')
            
            # Evaluar cada movimiento posible
            for movimiento in movimientos_legales:
                tablero.push(movimiento)
                valor = self._minimax(tablero, self.profundidad_maxima - 1, 
                                    float('-inf'), float('inf'), 
                                    not tablero.turn, inicio_tiempo)
                tablero.pop()
                
                # Verificar timeout
                if time.time() - inicio_tiempo > self.tiempo_limite:
                    logger.warning("Timeout en búsqueda Minimax")
                    break
                
                # Actualizar mejor movimiento
                if tablero.turn:  # Maximizar para blancas
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_movimiento = movimiento
                else:  # Minimizar para negras
                    if valor < mejor_valor:
                        mejor_valor = valor
                        mejor_movimiento = movimiento
            
            # Actualizar estadísticas
            tiempo_jugada = time.time() - inicio_tiempo
            self.estadisticas['tiempo_total'] += tiempo_jugada
            self.estadisticas['jugadas_realizadas'] += 1
            
            logger.debug("Minimax - Mejor jugada: %s, Valor: %.2f, Nodos: %d, Tiempo: %.3fs",
                        mejor_movimiento.uci() if mejor_movimiento else "None", 
                        mejor_valor, self.estadisticas['nodos_evaluados'], tiempo_jugada)
            
            return mejor_movimiento if mejor_movimiento is not None else movimientos_legales[0]
            
        except Exception as e:
            logger.error("Error en obtener_mejor_jugada Minimax: %s", str(e))
            return list(tablero.legal_moves)[0] if tablero.legal_moves else None
    
    def reiniciar_estadisticas(self):
        """Reinicia las estadísticas del agente."""
        self.estadisticas = {
            'nodos_evaluados': 0,
            'podas_alfa': 0,
            'podas_beta': 0,
            'tiempo_total': 0.0,
            'jugadas_realizadas': 0
        }
        logger.info("Estadísticas de Minimax reiniciadas")
    
    def __str__(self) -> str:
        return f"Minimax(profundidad={self.profundidad_maxima}, tiempo_limite={self.tiempo_limite}s)"