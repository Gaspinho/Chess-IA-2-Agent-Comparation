"""
Script para entrenar el agente PPO.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentes.agente_ppo import AgentePPO, CallbackEntrenamientoPPO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entrenamiento_ppo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def entrenar_ppo(args):
    """
    Entrena un agente PPO para ajedrez.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger.info("=== INICIANDO ENTRENAMIENTO PPO ===")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Modelo de salida: {args.output}")
    
    try:
        # Crear agente PPO con parámetros personalizados
        parametros_ppo = {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
            'max_grad_norm': args.max_grad_norm,
            'verbose': 1
        }
        
        agente = AgentePPO(**parametros_ppo)
        logger.info(f"Agente PPO creado con parámetros: {parametros_ppo}")
        
        # Crear callback para monitoreo
        callback = CallbackEntrenamientoPPO(verbose=1)
        
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
            
        if callback.pasos_por_episodio:
            pasos_promedio = sum(callback.pasos_por_episodio) / len(callback.pasos_por_episodio)
            logger.info(f"Pasos promedio por episodio: {pasos_promedio:.1f}")
        
        # Guardar estadísticas
        stats_file = args.output.replace('.zip', '_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Entrenamiento PPO - {datetime.now()}\n")
            f.write(f"Timesteps: {args.timesteps}\n")
            f.write(f"Duración: {duracion}\n")
            f.write(f"Episodios: {callback.episodios_completados}\n")
            if callback.recompensas_episodio:
                f.write(f"Recompensa promedio: {recompensa_promedio:.3f}\n")
                f.write(f"Mejor recompensa: {max(callback.recompensas_episodio):.3f}\n")
            if callback.pasos_por_episodio:
                f.write(f"Pasos promedio: {pasos_promedio:.1f}\n")
            f.write(f"Parámetros: {parametros_ppo}\n")
        
        logger.info(f"Estadísticas guardadas en {stats_file}")
        logger.info("=== ENTRENAMIENTO PPO COMPLETADO ===")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise


def continuar_entrenamiento(args):
    """
    Continúa el entrenamiento de un modelo PPO existente.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger.info("=== CONTINUANDO ENTRENAMIENTO PPO ===")
    logger.info(f"Modelo base: {args.model}")
    logger.info(f"Timesteps adicionales: {args.timesteps}")
    
    try:
        # Cargar modelo existente
        agente = AgentePPO(modelo_path=args.model)
        logger.info("Modelo cargado exitosamente")
        
        # Crear callback para monitoreo
        callback = CallbackEntrenamientoPPO(verbose=1)
        
        # Continuar entrenamiento
        logger.info("Continuando entrenamiento...")
        tiempo_inicio = datetime.now()
        
        agente.entrenar(
            total_timesteps=args.timesteps,
            callback=callback
        )
        
        tiempo_fin = datetime.now()
        duracion = tiempo_fin - tiempo_inicio
        logger.info(f"Entrenamiento adicional completado en {duracion}")
        
        # Guardar modelo actualizado
        nuevo_path = args.model.replace('.zip', f'_continued_{args.timesteps}.zip')
        agente.guardar_modelo(nuevo_path)
        logger.info(f"Modelo actualizado guardado en {nuevo_path}")
        
    except Exception as e:
        logger.error(f"Error continuando entrenamiento: {e}")
        raise


def probar_modelo(args):
    """
    Prueba un modelo PPO entrenado.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger.info("=== PROBANDO MODELO PPO ===")
    
    try:
        # Cargar modelo
        agente = AgentePPO(modelo_path=args.model)
        logger.info(f"Modelo cargado desde {args.model}")
        
        # Crear tablero de prueba
        import chess
        tablero = chess.Board()
        
        logger.info("Tablero inicial:")
        logger.info(str(tablero))
        
        # Probar algunos movimientos
        for i in range(args.movimientos):
            if tablero.is_game_over():
                logger.info(f"Juego terminado en movimiento {i+1}")
                break
                
            movimiento = agente.seleccionar_movimiento(tablero)
            if movimiento:
                logger.info(f"Movimiento {i+1}: {movimiento}")
                tablero.push(movimiento)
                
                if args.verbose:
                    logger.info(f"Tablero después del movimiento:")
                    logger.info(str(tablero))
                    logger.info(f"En jaque: {tablero.is_check()}")
                    logger.info(f"Movimientos legales: {len(list(tablero.legal_moves))}")
                    logger.info("-" * 40)
            else:
                logger.warning("No se pudo obtener movimiento")
                break
        
        # Estado final
        logger.info("=== ESTADO FINAL ===")
        logger.info(str(tablero))
        if tablero.is_checkmate():
            ganador = "Negras" if tablero.turn else "Blancas"
            logger.info(f"Jaque mate! Ganaron las {ganador}")
        elif tablero.is_stalemate():
            logger.info("Empate por ahogado")
        elif tablero.is_insufficient_material():
            logger.info("Empate por material insuficiente")
        else:
            logger.info("Juego en progreso")
        
        logger.info("=== PRUEBA COMPLETADA ===")
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {e}")
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar agente PPO para ajedrez')
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Subcomando para entrenar
    parser_entrenar = subparsers.add_parser('entrenar', help='Entrenar nuevo modelo')
    parser_entrenar.add_argument('--timesteps', type=int, default=100000,
                               help='Número de timesteps de entrenamiento')
    parser_entrenar.add_argument('--output', type=str, default='../modelos/ppo_modelo.zip',
                               help='Ruta de salida del modelo')
    parser_entrenar.add_argument('--learning_rate', type=float, default=3e-4,
                               help='Tasa de aprendizaje')
    parser_entrenar.add_argument('--n_steps', type=int, default=2048,
                               help='Número de pasos por actualización')
    parser_entrenar.add_argument('--batch_size', type=int, default=64,
                               help='Tamaño del batch')
    parser_entrenar.add_argument('--n_epochs', type=int, default=10,
                               help='Número de épocas por actualización')
    parser_entrenar.add_argument('--gamma', type=float, default=0.99,
                               help='Factor de descuento')
    parser_entrenar.add_argument('--gae_lambda', type=float, default=0.95,
                               help='Factor lambda para GAE')
    parser_entrenar.add_argument('--clip_range', type=float, default=0.2,
                               help='Rango de clipping para PPO')
    parser_entrenar.add_argument('--ent_coef', type=float, default=0.01,
                               help='Coeficiente de entropía')
    parser_entrenar.add_argument('--vf_coef', type=float, default=0.5,
                               help='Coeficiente de función de valor')
    parser_entrenar.add_argument('--max_grad_norm', type=float, default=0.5,
                               help='Norma máxima del gradiente')
    
    # Subcomando para continuar entrenamiento
    parser_continuar = subparsers.add_parser('continuar', help='Continuar entrenamiento existente')
    parser_continuar.add_argument('--model', type=str, required=True,
                                help='Ruta del modelo a continuar')
    parser_continuar.add_argument('--timesteps', type=int, default=50000,
                                help='Timesteps adicionales de entrenamiento')
    
    # Subcomando para probar
    parser_probar = subparsers.add_parser('probar', help='Probar modelo entrenado')
    parser_probar.add_argument('--model', type=str, required=True,
                             help='Ruta del modelo a probar')
    parser_probar.add_argument('--movimientos', type=int, default=10,
                             help='Número de movimientos a probar')
    parser_probar.add_argument('--verbose', action='store_true',
                             help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    if args.comando == 'entrenar':
        # Crear directorio de modelos si no existe
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        entrenar_ppo(args)
    elif args.comando == 'continuar':
        continuar_entrenamiento(args)
    elif args.comando == 'probar':
        probar_modelo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
