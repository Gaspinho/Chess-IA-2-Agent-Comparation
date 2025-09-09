# Configuración global para el proyecto de comparación de agentes de ajedrez

import logging
import os

# Configuración de logging
def configurar_logging(nivel=logging.INFO):
    """
    Configura el sistema de logging para todo el proyecto.
    """
    logging.basicConfig(
        level=nivel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('entrenamiento.log')
        ]
    )

# Rutas del proyecto
RUTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELOS = os.path.join(RUTA_RAIZ, 'modelos')
RUTA_RESULTADOS = os.path.join(RUTA_RAIZ, 'resultados')
RUTA_LOGS = os.path.join(RUTA_RAIZ, 'logs')

# Crear directorios si no existen
for ruta in [RUTA_MODELOS, RUTA_RESULTADOS, RUTA_LOGS]:
    os.makedirs(ruta, exist_ok=True)

# Configuración de los agentes
CONFIG_AGENTES = {
    'minimax': {
        'profundidad': 3,
        'usar_alpha_beta': True,
        'timeout': 10.0
    },
    'mcts': {
        'num_simulaciones': 1000,
        'c_puct': 1.414,
        'timeout': 10.0
    },
    'dqn': {
        'total_timesteps': 100000,
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.05,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000
    },
    'ppo': {
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    }
}

# Configuración de evaluación
CONFIG_EVALUACION = {
    'num_partidas': 50,
    'timeout_por_movimiento': 10.0,
    'guardar_partidas': True,
    'mostrar_progreso': True
}

# Configuración de análisis
CONFIG_ANALISIS = {
    'incluir_elo': True,
    'guardar_graficas': True,
    'formato_graficas': 'png',
    'dpi_graficas': 300,
    'estilo_graficas': 'seaborn-v0_8',
    'generar_reporte_html': True
}
