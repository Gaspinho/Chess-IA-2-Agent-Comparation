"""
Módulo para el entorno de ajedrez sin dependencias de pygame.

Este módulo proporciona una interfaz unificada para que tanto agentes de 
búsqueda como de aprendizaje por refuerzo puedan interactuar con el 
entorno de ajedrez usando python-chess directamente.
"""

import logging
import numpy as np
import chess
from typing import Optional, Tuple, Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntornoAjedrez:
    """
    Entorno de ajedrez sin dependencias de pygame.
    
    Esta clase proporciona métodos estándar que pueden ser utilizados tanto
    por agentes de búsqueda (Minimax, MCTS) como por agentes de RL (PPO).
    Usa python-chess directamente sin PettingZoo.
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Inicializa el entorno de ajedrez.
        
        Args:
            render_mode: Modo de renderizado (no usado, pero mantenido por compatibilidad)
        """
        self.render_mode = render_mode
        self.tablero = None
        self.agentes = ["player_0", "player_1"]  # Blancas y negras
        self.agente_actual = None
        self.observacion_actual = None
        self.recompensa_actual = 0.0
        self.terminado = False
        self.info_actual = {}
        self.historial_jugadas = []
        self._movimientos_legales_cache = []
        
        logger.info("EntornoAjedrez inicializado (sin pygame)")
    
    def reiniciar(self) -> np.ndarray:
        """
        Reinicia el entorno de ajedrez.
        
        Returns:
            Observación inicial del primer jugador (representación del tablero)
        """
        try:
            # Crear nuevo tablero
            self.tablero = chess.Board()
            self.agente_actual = "player_0"  # Comenzar con blancas
            self.terminado = False
            self.recompensa_actual = 0.0
            self.historial_jugadas = []
            self._movimientos_legales_cache = list(self.tablero.legal_moves)
            
            # Crear observación
            self.observacion_actual = self._tablero_a_observacion()
            self.info_actual = {
                'tablero': self.tablero,
                'turno': self.tablero.turn,
                'jugadas_legales': len(self._movimientos_legales_cache)
            }
            
            logger.info("Entorno reiniciado - Inicia jugador: %s", self.agente_actual)
            
            return self.observacion_actual
            
        except Exception as e:
            logger.error("Error al reiniciar el entorno: %s", str(e))
            raise
    
    def paso(self, accion: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            accion: Índice del movimiento en la lista de movimientos legales
            
        Returns:
            Tupla con (observación, recompensa, terminado, info)
        """
        try:
            if self.tablero is None:
                raise ValueError("El entorno no ha sido inicializado. Llamar reiniciar() primero.")
            
            if self.terminado:
                logger.warning("Intento de ejecutar acción en entorno terminado")
                return self.observacion_actual, 0.0, True, self.info_actual
            
            # Obtener movimientos legales actuales
            movimientos_legales = self._movimientos_legales_cache
            
            if accion >= len(movimientos_legales):
                logger.error(f"Acción inválida: {accion} >= {len(movimientos_legales)}")
                return self.observacion_actual, -10.0, True, self.info_actual
            
            # Ejecutar movimiento
            movimiento = movimientos_legales[accion]
            self.tablero.push(movimiento)
            self.historial_jugadas.append(movimiento)
            
            # Verificar estado del juego
            self.terminado = self.tablero.is_game_over()
            
            # Calcular recompensa
            if self.terminado:
                resultado = self.tablero.result()
                if resultado == "1-0":  # Blancas ganan
                    self.recompensa_actual = 1.0 if self.agente_actual == "player_0" else -1.0
                elif resultado == "0-1":  # Negras ganan
                    self.recompensa_actual = 1.0 if self.agente_actual == "player_1" else -1.0
                else:  # Empate
                    self.recompensa_actual = 0.0
            else:
                self.recompensa_actual = 0.0
            
            # Cambiar turno
            self.agente_actual = "player_1" if self.agente_actual == "player_0" else "player_0"
            
            # Actualizar cache de movimientos legales
            self._movimientos_legales_cache = list(self.tablero.legal_moves)
            
            # Crear nueva observación
            self.observacion_actual = self._tablero_a_observacion()
            
            # Actualizar info
            self.info_actual = {
                'tablero': self.tablero,
                'turno': self.tablero.turn,
                'jugadas_legales': len(self._movimientos_legales_cache),
                'ultimo_movimiento': movimiento,
                'es_jaque': self.tablero.is_check(),
                'es_jaque_mate': self.tablero.is_checkmate()
            }
            
            logger.debug("Paso ejecutado - Movimiento: %s, Recompensa: %.2f, Terminado: %s", 
                        movimiento, self.recompensa_actual, self.terminado)
            
            return (self.observacion_actual, 
                   self.recompensa_actual, 
                   self.terminado, 
                   self.info_actual)
            
        except Exception as e:
            logger.error("Error al ejecutar paso: %s", str(e))
            raise
    
    def _tablero_a_observacion(self) -> np.ndarray:
        """
        Convierte el tablero de chess a una representación numpy.
        
        Returns:
            Array numpy con representación del tablero (8x8x12)
            12 canales: 6 tipos de piezas x 2 colores
        """
        # Crear observación con 12 canales (6 piezas x 2 colores)
        obs = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Mapeo de piezas a índices
        piece_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.tablero.piece_at(square)
            if piece is not None:
                row = square // 8
                col = square % 8
                piece_idx = piece_to_idx[piece.piece_type]
                # Blancas en canales 0-5, negras en canales 6-11
                channel = piece_idx if piece.color == chess.WHITE else piece_idx + 6
                obs[row, col, channel] = 1.0
        
        return obs
    
    def obtener_acciones_legales(self) -> np.ndarray:
        """
        Obtiene las acciones legales para el jugador actual.
        
        Returns:
            Array con índices de acciones legales (índices en la lista de movimientos)
        """
        try:
            if self.tablero is None or self.terminado:
                return np.array([], dtype=np.int32)
            
            # Retornar índices de los movimientos legales
            num_movimientos = len(self._movimientos_legales_cache)
            acciones_legales = np.arange(num_movimientos, dtype=np.int32)
            
            return acciones_legales
            
        except Exception as e:
            logger.error("Error al obtener acciones legales: %s", str(e))
            return np.array([], dtype=np.int32)
    
    def obtener_tablero(self) -> chess.Board:
        """
        Obtiene el tablero de chess actual.
        
        Returns:
            Objeto chess.Board
        """
        return self.tablero
    
    def obtener_movimientos_legales(self) -> List[chess.Move]:
        """
        Obtiene la lista de movimientos legales del tablero.
        
        Returns:
            Lista de objetos chess.Move
        """
        return self._movimientos_legales_cache
    
    def renderizar(self):
        """Renderiza el estado actual del entorno (imprime el tablero)."""
        if self.tablero is not None:
            print(self.tablero)
            print(f"\nTurno: {'Blancas' if self.tablero.turn else 'Negras'}")
            print(f"Movimientos legales: {len(self._movimientos_legales_cache)}")
    
    def cerrar(self):
        """Cierra el entorno y libera recursos."""
        self.tablero = None
        self.agente_actual = None
        logger.info("Entorno cerrado correctamente")
    
    def obtener_estado_juego(self) -> Dict[str, Any]:
        """
        Obtiene el estado completo del juego.
        
        Returns:
            Diccionario con información del estado del juego
        """
        if self.tablero is None:
            return {'error': 'Tablero no inicializado'}
        
        return {
            'fen': self.tablero.fen(),
            'turno': 'Blancas' if self.tablero.turn == chess.WHITE else 'Negras',
            'es_jaque': self.tablero.is_check(),
            'es_jaque_mate': self.tablero.is_checkmate(),
            'es_ahogado': self.tablero.is_stalemate(),
            'material_insuficiente': self.tablero.is_insufficient_material(),
            'es_repeticion': self.tablero.is_repetition(),
            'jugadas_realizadas': len(self.historial_jugadas),
            'jugadas_legales': len(self._movimientos_legales_cache),
            'resultado': self.tablero.result() if self.tablero.is_game_over() else None
        }
    
    def __repr__(self) -> str:
        """Representación en string del entorno."""
        if self.tablero is None:
            return "EntornoAjedrez(no inicializado)"
        return f"EntornoAjedrez(turno={'Blancas' if self.tablero.turn else 'Negras'}, jugadas={len(self.historial_jugadas)})"
