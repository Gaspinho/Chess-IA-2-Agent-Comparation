"""
Módulo para envolver el entorno de ajedrez de PettingZoo.

Este módulo proporciona una interfaz unificada para que tanto agentes de 
búsqueda como de aprendizaje por refuerzo puedan interactuar con el 
entorno de ajedrez de PettingZoo.
"""

import logging
import numpy as np
import pettingzoo.classic.chess_v6 as chess_v6
from typing import Optional, Tuple, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntornoAjedrez:
    """
    Envuelve el entorno de ajedrez de PettingZoo para una interfaz unificada.
    
    Esta clase proporciona métodos estándar que pueden ser utilizados tanto
    por agentes de búsqueda (Minimax, MCTS) como por agentes de RL (DQN, PPO).
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Inicializa el entorno de ajedrez.
        
        Args:
            render_mode: Modo de renderizado ('human', 'rgb_array', None)
        """
        self.render_mode = render_mode
        self.env = None
        self.agentes = None
        self.agente_actual = None
        self.observacion_actual = None
        self.recompensa_actual = None
        self.terminado = False
        self.info_actual = None
        self.historial_jugadas = []
        
        logger.info("EntornoAjedrez inicializado")
    
    def reiniciar(self) -> np.ndarray:
        """
        Reinicia el entorno de ajedrez.
        
        Returns:
            Observación inicial del primer jugador
        """
        try:
            # Crear nuevo entorno
            self.env = chess_v6.env(render_mode=self.render_mode)
            self.env.reset()
            
            # Obtener lista de agentes
            self.agentes = self.env.agents
            self.agente_actual = self.agentes[0]  # Comenzar con blancas
            
            # Obtener observación inicial
            self.observacion_actual, self.recompensa_actual, \
            self.terminado, truncado, self.info_actual = self.env.last()
            
            # Limpiar historial
            self.historial_jugadas = []
            
            logger.info("Entorno reiniciado - Inicia jugador: %s", self.agente_actual)
            
            return self._normalizar_observacion(self.observacion_actual)
            
        except Exception as e:
            logger.error("Error al reiniciar el entorno: %s", str(e))
            raise
    
    def paso(self, accion: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            accion: Acción a ejecutar (índice de movimiento)
            
        Returns:
            Tupla con (observación, recompensa, terminado, info)
        """
        try:
            if self.env is None:
                raise ValueError("El entorno no ha sido inicializado. Llamar reiniciar() primero.")
            
            if self.terminado:
                logger.warning("Intento de ejecutar acción en entorno terminado")
                return self.observacion_actual, 0.0, True, self.info_actual
            
            # Ejecutar acción
            self.env.step(accion)
            
            # Obtener nueva observación
            self.observacion_actual, self.recompensa_actual, \
            self.terminado, truncado, self.info_actual = self.env.last()
            
            # Registrar jugada
            self.historial_jugadas.append({
                'agente': self.agente_actual,
                'accion': accion,
                'recompensa': self.recompensa_actual
            })
            
            # Cambiar al siguiente agente si no ha terminado
            if not self.terminado and not truncado:
                idx_actual = self.agentes.index(self.agente_actual)
                self.agente_actual = self.agentes[(idx_actual + 1) % len(self.agentes)]
            
            logger.debug("Paso ejecutado - Acción: %d, Recompensa: %.2f, Terminado: %s", 
                        accion, self.recompensa_actual, self.terminado)
            
            return (
                self._normalizar_observacion(self.observacion_actual),
                self._normalizar_recompensa(self.recompensa_actual),
                self.terminado or truncado,
                self.info_actual
            )
            
        except Exception as e:
            logger.error("Error en paso del entorno: %s", str(e))
            raise
    
    def obtener_acciones_legales(self) -> np.ndarray:
        """
        Obtiene las acciones legales para el jugador actual.
        
        Returns:
            Array con las acciones legales disponibles
        """
        try:
            if self.env is None or self.terminado:
                return np.array([])
            
            # Obtener máscara de acciones legales
            mascara_acciones = self.info_actual.get('action_mask', None)
            if mascara_acciones is not None:
                acciones_legales = np.where(mascara_acciones)[0]
            else:
                # Fallback: todas las acciones posibles
                # Verificar si action_space es una función o tiene el atributo n
                if hasattr(self.env.action_space, 'n'):
                    num_acciones = self.env.action_space.n
                elif callable(self.env.action_space):
                    # Si es una función, llamarla para obtener el espacio con el agente actual
                    if self.agente_actual:
                        action_space = self.env.action_space(self.agente_actual)
                    else:
                        # Si no hay agente actual, usar el primer agente disponible
                        action_space = self.env.action_space(self.agentes[0] if self.agentes else 'player_0')
                    num_acciones = action_space.n if hasattr(action_space, 'n') else 4096  # Default para chess
                else:
                    # Default para ajedrez (chess_v6 típicamente tiene 4096 acciones)
                    num_acciones = 4096
                acciones_legales = np.arange(num_acciones)
            
            return acciones_legales
            
        except Exception as e:
            logger.error("Error al obtener acciones legales: %s", str(e))
            return np.array([])
    
    def renderizar(self) -> Optional[np.ndarray]:
        """
        Renderiza el estado actual del entorno.
        
        Returns:
            Array de imagen si render_mode='rgb_array', None en caso contrario
        """
        try:
            if self.env is not None:
                return self.env.render()
            return None
            
        except Exception as e:
            logger.error("Error al renderizar: %s", str(e))
            return None
    
    def cerrar(self):
        """Cierra el entorno y libera recursos."""
        try:
            if self.env is not None:
                self.env.close()
                self.env = None
                logger.info("Entorno cerrado")
                
        except Exception as e:
            logger.error("Error al cerrar entorno: %s", str(e))
    
    def _normalizar_observacion(self, observacion) -> np.ndarray:
        """
        Normaliza la observación para mejorar el entrenamiento.
        
        Args:
            observacion: Observación raw del entorno (puede ser dict o array)
            
        Returns:
            Observación normalizada
        """
        if observacion is None:
            return np.zeros((8, 8, 111))  # Tamaño estándar para chess_v6
        
        # Manejar diferentes tipos de observación
        if isinstance(observacion, dict):
            # Si es un diccionario, buscar la observación principal
            if 'observation' in observacion:
                obs_array = observacion['observation']
            elif 'obs' in observacion:
                obs_array = observacion['obs']
            else:
                # Si no hay una clave estándar, tomar la primera entrada que sea un array
                for key, value in observacion.items():
                    if isinstance(value, np.ndarray):
                        obs_array = value
                        break
                else:
                    # Si no encontramos un array, crear uno por defecto
                    logger.warning("No se encontró observación válida en el diccionario, usando observación por defecto")
                    return np.zeros((8, 8, 111))
        else:
            # Si ya es un array, usarlo directamente
            obs_array = observacion
        
        # Asegurar que sea un array de NumPy
        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array)
        
        # La observación ya viene normalizada en chess_v6
        return obs_array.astype(np.float32)
    
    def _normalizar_recompensa(self, recompensa: float) -> float:
        """
        Normaliza la recompensa.
        
        Args:
            recompensa: Recompensa raw del entorno
            
        Returns:
            Recompensa normalizada
        """
        # Las recompensas en chess_v5 ya están en rango [-1, 1]
        return float(recompensa)
    
    @property
    def espacio_observacion(self):
        """Retorna el espacio de observación del entorno."""
        if self.env is not None:
            if hasattr(self.env, 'observation_space'):
                if callable(self.env.observation_space):
                    # Usar el agente actual o el primer agente disponible
                    agente = self.agente_actual if self.agente_actual else (self.agentes[0] if self.agentes else 'player_0')
                    return self.env.observation_space(agente)
                else:
                    return self.env.observation_space
        return None
    
    @property
    def espacio_accion(self):
        """Retorna el espacio de acción del entorno."""
        if self.env is not None:
            if hasattr(self.env, 'action_space'):
                if callable(self.env.action_space):
                    # Usar el agente actual o el primer agente disponible
                    agente = self.agente_actual if self.agente_actual else (self.agentes[0] if self.agentes else 'player_0')
                    return self.env.action_space(agente)
                else:
                    return self.env.action_space
        return None
    
    @property
    def es_terminado(self) -> bool:
        """Retorna True si el entorno ha terminado."""
        return self.terminado
    
    @property
    def jugador_actual(self) -> str:
        """Retorna el identificador del jugador actual."""
        return self.agente_actual
    
    def obtener_resultado(self) -> Dict[str, Any]:
        """
        Obtiene el resultado final del juego.
        
        Returns:
            Diccionario con información del resultado
        """
        if not self.terminado:
            return {
                'estado': 'en_progreso',
                'ganador': None,
                'tipo_fin': 'en_progreso',
                'total_jugadas': len(self.historial_jugadas)
            }
        
        # Determinar ganador basado en las recompensas finales
        resultado = {
            'estado': 'terminado',
            'ganador': None,
            'tipo_fin': 'desconocido',
            'total_jugadas': len(self.historial_jugadas)
        }
        
        if self.recompensa_actual == 1:
            resultado['ganador'] = self.agente_actual
            resultado['tipo_fin'] = 'victoria'
        elif self.recompensa_actual == -1:
            # El oponente ganó
            idx_actual = self.agentes.index(self.agente_actual)
            resultado['ganador'] = self.agentes[(idx_actual + 1) % len(self.agentes)]
            resultado['tipo_fin'] = 'derrota'
        else:
            resultado['tipo_fin'] = 'empate'
        
        return resultado
    
    def obtener_estado_tablero(self) -> str:
        """
        Obtiene representación en string del estado del tablero.
        
        Returns:
            String representando el estado del tablero
        """
        try:
            if hasattr(self.env, 'board'):
                return str(self.env.board)
            return "Estado del tablero no disponible"
            
        except Exception as e:
            logger.error("Error al obtener estado del tablero: %s", str(e))
            return "Error al obtener estado"