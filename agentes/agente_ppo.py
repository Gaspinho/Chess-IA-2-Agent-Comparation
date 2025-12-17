"""
Agente PPO (Proximal Policy Optimization) para ajedrez.

Este módulo implementa un agente de aprendizaje profundo por refuerzo
que utiliza el algoritmo PPO para aprender a jugar ajedrez.
"""

import logging
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessGymWrapper(gym.Env):
    """Wrapper para convertir EntornoAjedrez a formato Gymnasium."""
    
    def __init__(self, entorno_base):
        super(ChessGymWrapper, self).__init__()
        self.entorno = entorno_base
        
        # Definir espacios de observación y acción
        # Nuestro entorno usa (8, 8, 12): 12 canales para 6 tipos de piezas x 2 colores
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.float32
        )
        
        # Espacio de acciones: dinámico basado en movimientos legales
        # Usamos un espacio grande para compatibilidad
        self.action_space = spaces.Discrete(218)  # Máximo de movimientos legales aproximado
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.entorno.reiniciar()
        return obs, {}
    
    def step(self, action):
        # Verificar si la acción es legal
        acciones_legales = self.entorno.obtener_acciones_legales()
        
        if len(acciones_legales) == 0:
            # No hay movimientos legales, juego terminado
            return self.entorno.observacion_actual, 0.0, True, False, {'error': 'no_legal_moves'}
        
        # Mapear acción al rango de movimientos legales si es necesario
        if action >= len(acciones_legales):
            # Si la acción está fuera de rango, elegir una legal aleatoria
            action = np.random.choice(acciones_legales)
        
        obs, reward, terminated, info = self.entorno.paso(action)
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        if hasattr(self.entorno, 'cerrar'):
            self.entorno.cerrar()


class ProgressCallback(BaseCallback):
    """Callback para mostrar progreso durante el entrenamiento."""
    
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            logger.info(f"Rollout completado - Timesteps: {self.num_timesteps}")


class AgentePPO:
    """
    Agente que utiliza Proximal Policy Optimization para jugar ajedrez.
    
    Este agente utiliza redes neuronales profundas para aprender políticas
    de juego óptimas mediante entrenamiento con aprendizaje por refuerzo.
    """
    
    def __init__(self, env=None, learning_rate: float = 3e-4, 
                 n_steps: int = 2048, batch_size: int = 64,
                 n_epochs: int = 10, gamma: float = 0.99,
                 device: str = 'auto'):
        """
        Inicializa el agente PPO.
        
        Args:
            env: Entorno de ajedrez (si es None, se creará después)
            learning_rate: Tasa de aprendizaje
            n_steps: Pasos por actualización de política
            batch_size: Tamaño del batch para entrenamiento
            n_epochs: Épocas de optimización
            gamma: Factor de descuento
            device: Device para PyTorch ('auto', 'cpu', 'cuda')
        """
        self.nombre = "PPO"
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.device = device
        
        self.modelo = None
        self.env = env
        
        self.estadisticas = {
            'pasos_entrenamiento': 0,
            'episodios_completados': 0,
            'recompensa_promedio': 0.0,
            'tiempo_entrenamiento': 0.0,
            'jugadas_realizadas': 0
        }
        
        logger.info("Agente PPO inicializado - LR: %.2e, Steps: %d, Batch: %d",
                   learning_rate, n_steps, batch_size)
    
    def crear_modelo(self, env):
        """
        Crea el modelo PPO con el entorno especificado.
        
        Args:
            env: Entorno de entrenamiento
        """
        # Envolver en DummyVecEnv si no está vectorizado
        if not hasattr(env, 'num_envs'):
            env = DummyVecEnv([lambda: env])
        
        # Crear modelo PPO con política simplificada
        # Usar MlpPolicy para tableros de ajedrez
        self.modelo = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            verbose=0,  # Cambiar a 0 para menos output
            device=self.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128])  # Redes más pequeñas para chess
            )
        )
        
        self.env = env
        logger.info("Modelo PPO creado exitosamente con observaciones (8,8,12)")
    
    def entrenar(self, total_timesteps: int = 100000, progreso: bool = True):
        """
        Entrena el agente PPO.
        
        Args:
            total_timesteps: Número total de pasos de entrenamiento
            progreso: Si mostrar barra de progreso
        """
        if self.modelo is None:
            raise ValueError("Modelo no inicializado. Llamar crear_modelo() primero.")
        
        inicio = time.time()
        
        callback = ProgressCallback(verbose=1) if progreso else None
        
        logger.info("Iniciando entrenamiento PPO - Total timesteps: %d", total_timesteps)
        
        try:
            self.modelo.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=progreso
            )
            
            tiempo_entrenamiento = time.time() - inicio
            self.estadisticas['pasos_entrenamiento'] += total_timesteps
            self.estadisticas['tiempo_entrenamiento'] += tiempo_entrenamiento
            
            logger.info("Entrenamiento completado - Tiempo: %.2fs", tiempo_entrenamiento)
            
        except Exception as e:
            logger.error("Error durante el entrenamiento: %s", str(e))
            raise
    
    def seleccionar_accion(self, observacion: np.ndarray, acciones_legales: np.ndarray,
                          info: Dict[str, Any]) -> int:
        """
        Selecciona una acción usando el modelo PPO entrenado.
        
        Args:
            observacion: Estado actual del juego
            acciones_legales: Acciones legales disponibles
            info: Información adicional del entorno
            
        Returns:
            Índice de la acción seleccionada
        """
        inicio_tiempo = time.time()
        
        try:
            if len(acciones_legales) == 0:
                logger.warning("No hay acciones legales disponibles")
                return 0
            
            if len(acciones_legales) == 1:
                return acciones_legales[0]
            
            if self.modelo is None:
                # Si no hay modelo entrenado, selección aleatoria
                logger.warning("Modelo no entrenado, selección aleatoria")
                return np.random.choice(acciones_legales)
            
            # Predecir acción con el modelo
            accion, _states = self.modelo.predict(observacion, deterministic=True)
            
            # Verificar si la acción es legal
            if accion not in acciones_legales:
                # Si no es legal, elegir la más cercana de las legales
                logger.debug("Acción predicha no es legal, seleccionando alternativa")
                accion = acciones_legales[0]
            
            self.estadisticas['jugadas_realizadas'] += 1
            
            return int(accion)
            
        except Exception as e:
            logger.error("Error en selección de acción PPO: %s", str(e))
            return np.random.choice(acciones_legales)
    
    def guardar_modelo(self, ruta: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            ruta: Ruta donde guardar el modelo
        """
        if self.modelo is None:
            raise ValueError("No hay modelo para guardar")
        
        self.modelo.save(ruta)
        logger.info("Modelo guardado en: %s", ruta)
    
    def cargar_modelo(self, ruta: str):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            ruta: Ruta del modelo a cargar
        """
        try:
            self.modelo = PPO.load(ruta, device=self.device)
            logger.info("Modelo cargado desde: %s", ruta)
        except Exception as e:
            logger.error("Error al cargar modelo: %s", str(e))
            raise
    
    def evaluar(self, env, n_episodios: int = 10) -> Dict[str, float]:
        """
        Evalúa el agente en el entorno.
        
        Args:
            env: Entorno de evaluación
            n_episodios: Número de episodios para evaluar
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.modelo is None:
            raise ValueError("Modelo no inicializado")
        
        recompensas = []
        longitudes = []
        
        for _ in range(n_episodios):
            obs, _ = env.reset()
            terminado = False
            recompensa_episodio = 0
            longitud = 0
            
            while not terminado:
                accion, _ = self.modelo.predict(obs, deterministic=True)
                obs, reward, terminado, truncado, _ = env.step(accion)
                recompensa_episodio += reward
                longitud += 1
                
                if truncado:
                    break
            
            recompensas.append(recompensa_episodio)
            longitudes.append(longitud)
        
        metricas = {
            'recompensa_promedio': np.mean(recompensas),
            'recompensa_std': np.std(recompensas),
            'longitud_promedio': np.mean(longitudes),
            'longitud_std': np.std(longitudes)
        }
        
        logger.info("Evaluación completada - Recompensa promedio: %.2f", 
                   metricas['recompensa_promedio'])
        
        return metricas
    
    def obtener_info_modelo(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        if self.modelo is None:
            return {"estado": "no_inicializado"}
        
        return {
            "estado": "inicializado",
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "device": self.device,
            "pasos_entrenamiento": self.estadisticas['pasos_entrenamiento']
        }


def crear_agente_ppo(env, learning_rate: float = 3e-4, 
                    n_steps: int = 2048, batch_size: int = 64,
                    n_epochs: int = 10) -> AgentePPO:
    """
    Función auxiliar para crear un agente PPO completamente configurado.
    
    Args:
        env: Entorno de ajedrez
        learning_rate: Tasa de aprendizaje
        n_steps: Pasos por actualización
        batch_size: Tamaño del batch
        n_epochs: Épocas de optimización
        
    Returns:
        Agente PPO configurado
    """
    # Envolver entorno si es necesario
    if not isinstance(env, gym.Env):
        from entorno.entorno_ajedrez import EntornoAjedrez
        if isinstance(env, EntornoAjedrez):
            env = ChessGymWrapper(env)
    
    agente = AgentePPO(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs
    )
    
    agente.crear_modelo(env)
    
    return agente
