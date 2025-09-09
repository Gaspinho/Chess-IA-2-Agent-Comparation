"""
Módulo del entorno de ajedrez para agentes IA.
Configura y maneja el entorno de PettingZoo Chess.
"""

import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
import chess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntornoAjedrez:
    """
    Wrapper para el entorno de ajedrez de PettingZoo.
    Proporciona interfaz unificada para agentes RL y de búsqueda.
    """
    
    def __init__(self, render_mode=None):
        """
        Inicializa el entorno de ajedrez.
        
        Args:
            render_mode: Modo de renderizado ('human', 'ansi', None)
        """
        self.env = chess_v6.env(render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.current_agent = None
        self.board = None
        self.game_over = False
        
    def reiniciar(self):
        """
        Reinicia el entorno para una nueva partida.
        
        Returns:
            tuple: (observaciones, info) para el primer agente
        """
        observations, infos = self.env.reset()
        self.current_agent = self.env.agent_selection
        self.game_over = False
        
        # Obtener el tablero actual
        if hasattr(self.env.unwrapped, 'board'):
            self.board = self.env.unwrapped.board
        else:
            self.board = chess.Board()
            
        logger.info("Partida reiniciada")
        return observations, infos
    
    def paso(self, accion):
        """
        Ejecuta un paso en el entorno.
        
        Args:
            accion: Acción a ejecutar
            
        Returns:
            tuple: (observacion, recompensa, terminado, truncado, info)
        """
        if self.game_over:
            logger.warning("Intentando ejecutar acción en juego terminado")
            return None, 0, True, False, {}
            
        # Ejecutar la acción
        self.env.step(accion)
        
        # Obtener el estado después del paso
        observacion = self.env.observe(self.env.agent_selection)
        recompensa = self.env.rewards.get(self.current_agent, 0)
        terminado = self.env.terminations.get(self.env.agent_selection, False)
        truncado = self.env.truncations.get(self.env.agent_selection, False)
        
        # Actualizar el agente actual
        self.current_agent = self.env.agent_selection
        
        # Actualizar el tablero
        if hasattr(self.env.unwrapped, 'board'):
            self.board = self.env.unwrapped.board
            
        # Verificar si el juego terminó
        self.game_over = terminado or truncado or not self.env.agents
        
        info = {
            'agente_actual': self.current_agent,
            'movimientos_legales': self.obtener_movimientos_legales(),
            'juego_terminado': self.game_over
        }
        
        return observacion, recompensa, terminado, truncado, info
    
    def obtener_movimientos_legales(self):
        """
        Obtiene los movimientos legales disponibles.
        
        Returns:
            list: Lista de movimientos legales
        """
        if self.board is None:
            return []
            
        try:
            # Obtener máscara de acciones legales
            mask = self.env.action_mask(self.env.agent_selection)
            movimientos_legales = [i for i, legal in enumerate(mask) if legal]
            return movimientos_legales
        except Exception as e:
            logger.error(f"Error obteniendo movimientos legales: {e}")
            return []
    
    def obtener_observacion(self, agente=None):
        """
        Obtiene la observación actual para un agente.
        
        Args:
            agente: Nombre del agente (si None, usa el agente actual)
            
        Returns:
            np.array: Observación del estado actual
        """
        if agente is None:
            agente = self.env.agent_selection
            
        try:
            return self.env.observe(agente)
        except Exception as e:
            logger.error(f"Error obteniendo observación: {e}")
            return np.zeros(self.observation_space.shape)
    
    def renderizar(self):
        """Renderiza el estado actual del juego."""
        try:
            return self.env.render()
        except Exception as e:
            logger.error(f"Error renderizando: {e}")
            if self.board:
                print(self.board)
    
    def cerrar(self):
        """Cierra el entorno."""
        try:
            self.env.close()
            logger.info("Entorno cerrado correctamente")
        except Exception as e:
            logger.error(f"Error cerrando entorno: {e}")
    
    def obtener_estado_juego(self):
        """
        Obtiene el estado actual del juego.
        
        Returns:
            dict: Diccionario con información del estado
        """
        estado = {
            'agente_actual': self.current_agent,
            'juego_terminado': self.game_over,
            'tablero': str(self.board) if self.board else None,
            'turno_blancas': self.board.turn if self.board else True,
            'en_jaque': self.board.is_check() if self.board else False,
            'jaque_mate': self.board.is_checkmate() if self.board else False,
            'empate': self.board.is_stalemate() if self.board else False
        }
        return estado
    
    def es_movimiento_legal(self, accion):
        """
        Verifica si una acción es legal.
        
        Args:
            accion: Acción a verificar
            
        Returns:
            bool: True si la acción es legal
        """
        try:
            mask = self.env.action_mask(self.env.agent_selection)
            return mask[accion] if accion < len(mask) else False
        except Exception as e:
            logger.error(f"Error verificando legalidad: {e}")
            return False
    
    def obtener_recompensa_actual(self, agente=None):
        """
        Obtiene la recompensa actual para un agente.
        
        Args:
            agente: Nombre del agente
            
        Returns:
            float: Recompensa actual
        """
        if agente is None:
            agente = self.current_agent
            
        return self.env.rewards.get(agente, 0)
    
    def copiar_tablero(self):
        """
        Crea una copia del tablero actual.
        
        Returns:
            chess.Board: Copia del tablero
        """
        if self.board:
            return self.board.copy()
        return chess.Board()


def crear_entorno_vectorizado(num_envs=1, render_mode=None):
    """
    Crea múltiples entornos vectorizados para entrenamiento paralelo.
    
    Args:
        num_envs: Número de entornos
        render_mode: Modo de renderizado
        
    Returns:
        VecEnv: Entorno vectorizado
    """
    def make_env():
        def _init():
            return EntornoAjedrez(render_mode=render_mode)
        return _init
    
    # Para stable-baselines3, necesitamos un wrapper específico
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    if num_envs == 1:
        return DummyVecEnv([make_env()])
    else:
        return SubprocVecEnv([make_env() for _ in range(num_envs)])


if __name__ == "__main__":
    # Prueba básica del entorno
    print("Probando entorno de ajedrez...")
    
    entorno = EntornoAjedrez(render_mode="ansi")
    obs, info = entorno.reiniciar()
    
    print("Estado inicial:")
    entorno.renderizar()
    
    print(f"Agente actual: {entorno.current_agent}")
    print(f"Movimientos legales disponibles: {len(entorno.obtener_movimientos_legales())}")
    
    entorno.cerrar()
    print("Prueba completada")
