"""
Script para entrenar el agente DQN.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentes.agente_dqn import AgenteDQN, CallbackEntrenamiento

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entrenamiento_dqn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def entrenar_dqn(args):
    """
    Entrena un agente DQN para ajedrez.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger.info("=== INICIANDO ENTRENAMIENTO DQN ===")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Modelo de salida: {args.output}")
    
    try:
        # Crear agente DQN con parámetros personalizados
        parametros_dqn = {
            'learning_rate': args.learning_rate,
            'buffer_size': args.buffer_size,
            'batch_size': args.batch_size,
            'gamma': args.gamma,
            'target_update_interval': args.target_update_interval,
            'exploration_fraction': args.exploration_fraction,
            'exploration_initial_eps': args.exploration_initial_eps,
            'exploration_final_eps': args.exploration_final_eps,
            'train_freq': args.train_freq,
            'verbose': 1
        }
        
        agente = AgenteDQN(**parametros_dqn)
        logger.info(f"Agente DQN creado con parámetros: {parametros_dqn}")
        
        # Crear callback para monitoreo
        callback = CallbackEntrenamiento(verbose=1)
        
        # Entrenar el modelo
        logger.info("Iniciando entrenamiento...")
        tiempo_inicio = datetime.now()
        
        agente.entrenar(
            total_timesteps=args.timesteps,
            callback=callback
        )
        
        tiempo_fin = datetime.now()
        duracion = tiempo_fin - tiempo_inicio
        logger.info(f"Entrenamiento completado en {duracion}")
        
        # Guardar el modelo
        agente.guardar_modelo(args.output)
        
        # Mostrar estadísticas finales
        logger.info("=== ESTADÍSTICAS FINALES ===")
        logger.info(f"Episodios completados: {callback.episodios_completados}")
        if callback.recompensas_episodio:
            recompensa_promedio = sum(callback.recompensas_episodio) / len(callback.recompensas_episodio)
            logger.info(f"Recompensa promedio: {recompensa_promedio:.3f}")
            logger.info(f"Mejor recompensa: {max(callback.recompensas_episodio):.3f}")
            logger.info(f"Peor recompensa: {min(callback.recompensas_episodio):.3f}")
        
        # Guardar estadísticas
        stats_file = args.output.replace('.zip', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Entrenamiento DQN - {datetime.now()}\n")
            f.write(f"Timesteps: {args.timesteps}\n")
            f.write(f"Duración: {duracion}\n")
            f.write(f"Episodios: {callback.episodios_completados}\n")
            if callback.recompensas_episodio:
                f.write(f"Recompensa promedio: {recompensa_promedio:.3f}\n")
                f.write(f"Mejor recompensa: {max(callback.recompensas_episodio):.3f}\n")
            f.write(f"Parámetros: {parametros_dqn}\n")
        
        logger.info(f"Estadísticas guardadas en {stats_file}")
        logger.info("=== ENTRENAMIENTO DQN COMPLETADO ===")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise


def probar_modelo(args):
    """
    Prueba un modelo DQN entrenado.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger.info("=== PROBANDO MODELO DQN ===")
    
    try:
        # Cargar modelo
        agente = AgenteDQN(modelo_path=args.model)
        logger.info(f"Modelo cargado desde {args.model}")
        
        # Crear tablero de prueba
        import chess
        tablero = chess.Board()
        
        logger.info("Tablero inicial:")
        logger.info(str(tablero))
        
        # Probar algunos movimientos
        for i in range(5):
            if tablero.is_game_over():
                break
                
            movimiento = agente.seleccionar_movimiento(tablero)
            if movimiento:
                logger.info(f"Movimiento {i+1}: {movimiento}")
                tablero.push(movimiento)
                logger.info(f"Tablero después del movimiento:")
                logger.info(str(tablero))
            else:
                logger.warning("No se pudo obtener movimiento")
                break
        
        logger.info("=== PRUEBA COMPLETADA ===")
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {e}")
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar agente DQN para ajedrez')
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Subcomando para entrenar
    parser_entrenar = subparsers.add_parser('entrenar', help='Entrenar nuevo modelo')
    parser_entrenar.add_argument('--timesteps', type=int, default=100000,
                               help='Número de timesteps de entrenamiento')
    parser_entrenar.add_argument('--output', type=str, default='../modelos/dqn_modelo.zip',
                               help='Ruta de salida del modelo')
    parser_entrenar.add_argument('--learning_rate', type=float, default=1e-4,
                               help='Tasa de aprendizaje')
    parser_entrenar.add_argument('--buffer_size', type=int, default=50000,
                               help='Tamaño del buffer de experiencia')
    parser_entrenar.add_argument('--batch_size', type=int, default=64,
                               help='Tamaño del batch')
    parser_entrenar.add_argument('--gamma', type=float, default=0.99,
                               help='Factor de descuento')
    parser_entrenar.add_argument('--target_update_interval', type=int, default=1000,
                               help='Intervalo de actualización de red objetivo')
    parser_entrenar.add_argument('--exploration_fraction', type=float, default=0.1,
                               help='Fracción de entrenamiento para exploración')
    parser_entrenar.add_argument('--exploration_initial_eps', type=float, default=1.0,
                               help='Epsilon inicial para exploración')
    parser_entrenar.add_argument('--exploration_final_eps', type=float, default=0.05,
                               help='Epsilon final para exploración')
    parser_entrenar.add_argument('--train_freq', type=int, default=4,
                               help='Frecuencia de entrenamiento')
    
    # Subcomando para probar
    parser_probar = subparsers.add_parser('probar', help='Probar modelo entrenado')
    parser_probar.add_argument('--model', type=str, required=True,
                             help='Ruta del modelo a probar')
    
    args = parser.parse_args()
    
    if args.comando == 'entrenar':
        # Crear directorio de modelos si no existe
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        entrenar_dqn(args)
    elif args.comando == 'probar':
        probar_modelo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
