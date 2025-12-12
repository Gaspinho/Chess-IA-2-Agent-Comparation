"""
Archivo principal del proyecto Chess-IA-2-Agent-Comparation.

Este script proporciona una interfaz de línea de comandos para entrenar,
evaluar y analizar agentes de ajedrez.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def entrenar_alphazero(args) -> Dict[str, Any]:
    """Entrena el agente AlphaZero."""
    logger.info("=== ENTRENANDO AGENTE ALPHAZERO ===")
    
    try:
        from entrenamiento.entrenar_alphazero import entrenar_agente_alphazero
        
        config = {
            'learning_rate': args.learning_rate,
            'num_simulaciones': args.num_simulaciones,
            'batch_size': args.batch_size,
            'num_iteraciones': args.num_iteraciones,
            'num_episodios': args.num_episodios,
            'temperatura': args.temperatura,
            'c_puct': args.c_puct,
            'frecuencia_guardado': args.save_freq,
            'ruta_guardado': 'modelos/'
        }
        
        resultados = entrenar_agente_alphazero(config)
        
        if resultados.get('entrenamiento_exitoso', False):
            logger.info("Entrenamiento AlphaZero completado exitosamente")
            return resultados
        else:
            logger.error("Error en entrenamiento AlphaZero: %s", resultados.get('error'))
            return resultados
            
    except Exception as e:
        logger.error("Error al entrenar AlphaZero: %s", str(e))
        return {'entrenamiento_exitoso': False, 'error': str(e)}


def entrenar_ppo(args) -> Dict[str, Any]:
    """Entrena el agente PPO."""
    logger.info("=== ENTRENANDO AGENTE PPO ===")
    
    try:
        from entrenamiento.entrenar_ppo import entrenar_agente_ppo
        
        config = {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'total_timesteps': args.timesteps,
            'frecuencia_guardado': args.save_freq,
            'ruta_guardado': 'modelos/'
        }
        
        resultados = entrenar_agente_ppo(config)
        
        if resultados.get('entrenamiento_exitoso', False):
            logger.info("Entrenamiento PPO completado exitosamente")
            return resultados
        else:
            logger.error("Error en entrenamiento PPO: %s", resultados.get('error'))
            return resultados
            
    except Exception as e:
        logger.error("Error al entrenar PPO: %s", str(e))
        return {'entrenamiento_exitoso': False, 'error': str(e)}


def evaluar_agentes(args) -> Dict[str, Any]:
    """Evalúa todos los agentes."""
    logger.info("=== EVALUANDO AGENTES ===")
    
    try:
        from entrenamiento.evaluar_agentes import EvaluadorAgentes
        
        evaluador = EvaluadorAgentes(
            timeout_jugada=args.timeout,
            max_jugadas=args.max_moves
        )
        
        resultados = evaluador.ejecutar_evaluacion_completa()
        
        logger.info("Evaluación completada: %d partidas jugadas", 
                   resultados['total_partidas'])
        
        return resultados
        
    except Exception as e:
        logger.error("Error al evaluar agentes: %s", str(e))
        return {'evaluacion_exitosa': False, 'error': str(e)}


def generar_reporte(args) -> bool:
    """Genera el reporte de análisis."""
    logger.info("=== GENERANDO REPORTE ===")
    
    try:
        # Verificar que existan resultados
        archivo_resultados = 'resultados/comparacion_agentes.csv'
        if not os.path.exists(archivo_resultados):
            logger.error("No se encontraron resultados para analizar. Ejecutar evaluación primero.")
            return False
        
        from analisis.generar_reporte import GeneradorReporte
        
        generador = GeneradorReporte()
        exito = generador.generar_reporte_completo(
            archivo_datos=archivo_resultados,
            archivo_salida='analisis/reporte.md'
        )
        
        if exito:
            logger.info("Reporte generado exitosamente en: analisis/reporte.md")
        else:
            logger.error("Error al generar reporte")
        
        return exito
        
    except Exception as e:
        logger.error("Error al generar reporte: %s", str(e))
        return False


def crear_parser() -> argparse.ArgumentParser:
    """Crea el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Sistema de entrenamiento y evaluación de agentes de ajedrez',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python principal.py --entrenar alphazero --num-iteraciones 100
  python principal.py --entrenar ppo --timesteps 100000 --learning-rate 0.0001
  python principal.py --evaluar --timeout 15 --max-moves 300
  python principal.py --reporte
  python principal.py --entrenar alphazero --evaluar --reporte
        """
    )
    
    # Argumentos principales
    parser.add_argument('--entrenar', choices=['alphazero', 'ppo'], 
                       help='Entrenar un agente específico')
    parser.add_argument('--evaluar', action='store_true',
                       help='Evaluar todos los agentes')
    parser.add_argument('--reporte', action='store_true',
                       help='Generar reporte de análisis')
    
    # Argumentos de entrenamiento
    training_group = parser.add_argument_group('Parámetros de entrenamiento')
    training_group.add_argument('--timesteps', type=int, default=100000,
                               help='Número de timesteps para entrenamiento (default: 100000)')
    training_group.add_argument('--learning-rate', type=float, default=3e-4,
                               help='Tasa de aprendizaje (default: 3e-4)')
    training_group.add_argument('--batch-size', type=int, default=64,
                               help='Tamaño del batch (default: 64)')
    training_group.add_argument('--gamma', type=float, default=0.99,
                               help='Factor de descuento (default: 0.99)')
    training_group.add_argument('--save-freq', type=int, default=10000,
                               help='Frecuencia de guardado (default: 10000)')
    
    # Argumentos específicos para PPO
    ppo_group = parser.add_argument_group('Parámetros específicos de PPO')
    ppo_group.add_argument('--n-steps', type=int, default=2048,
                          help='Número de pasos para PPO (default: 2048)')
    ppo_group.add_argument('--n-epochs', type=int, default=10,
                          help='Número de épocas para PPO (default: 10)')
    
    # Argumentos específicos para AlphaZero
    alphazero_group = parser.add_argument_group('Parámetros específicos de AlphaZero')
    alphazero_group.add_argument('--num-simulaciones', type=int, default=800,
                          help='Número de simulaciones MCTS para AlphaZero (default: 800)')
    alphazero_group.add_argument('--num-iteraciones', type=int, default=100,
                          help='Número de iteraciones de auto-juego (default: 100)')
    alphazero_group.add_argument('--num-episodios', type=int, default=100,
                          help='Episodios por iteración (default: 100)')
    alphazero_group.add_argument('--temperatura', type=float, default=1.0,
                          help='Temperatura para exploración (default: 1.0)')
    alphazero_group.add_argument('--c-puct', type=float, default=1.5,
                          help='Constante de exploración PUCT (default: 1.5)')
    
    # Argumentos de evaluación
    eval_group = parser.add_argument_group('Parámetros de evaluación')
    eval_group.add_argument('--timeout', type=float, default=10.0,
                           help='Timeout por jugada en segundos (default: 10.0)')
    eval_group.add_argument('--max-moves', type=int, default=500,
                           help='Máximo número de jugadas por partida (default: 500)')
    
    # Argumentos generales
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Activar logging detallado')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Silenciar output no esencial')
    
    return parser


def verificar_dependencias():
    """Verifica que las dependencias estén instaladas."""
    dependencias_requeridas = [
        'pettingzoo', 'stable_baselines3', 'torch', 
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'chess'
    ]
    
    dependencias_faltantes = []
    
    for dep in dependencias_requeridas:
        try:
            __import__(dep)
        except ImportError:
            dependencias_faltantes.append(dep)
    
    if dependencias_faltantes:
        logger.error("Dependencias faltantes: %s", ', '.join(dependencias_faltantes))
        logger.error("Instalar con: pip install -r requerimientos.txt")
        return False
    
    return True


def crear_directorios():
    """Crea los directorios necesarios."""
    directorios = ['modelos', 'resultados', 'analisis']
    
    for directorio in directorios:
        os.makedirs(directorio, exist_ok=True)
        logger.debug("Directorio creado/verificado: %s", directorio)


def main():
    """Función principal."""
    # Crear parser y parsear argumentos
    parser = crear_parser()
    args = parser.parse_args()
    
    # Configurar logging basado en argumentos
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Mostrar información inicial
    logger.info("=== CHESS-IA-2-AGENT-COMPARATION ===")
    logger.info("Sistema de entrenamiento y evaluación de agentes de ajedrez")
    
    # Verificar dependencias
    if not verificar_dependencias():
        sys.exit(1)
    
    # Crear directorios necesarios
    crear_directorios()
    
    # Verificar que al menos una acción esté especificada
    if not (args.entrenar or args.evaluar or args.reporte):
        logger.error("Debe especificar al menos una acción: --entrenar, --evaluar, o --reporte")
        parser.print_help()
        sys.exit(1)
    
    exitos = []
    
    # Ejecutar entrenamiento si se especifica
    if args.entrenar:
        if args.entrenar == 'alphazero':
            resultado = entrenar_alphazero(args)
            exitos.append(resultado.get('entrenamiento_exitoso', False))
        elif args.entrenar == 'ppo':
            resultado = entrenar_ppo(args)
            exitos.append(resultado.get('entrenamiento_exitoso', False))
    
    # Ejecutar evaluación si se especifica
    if args.evaluar:
        resultado = evaluar_agentes(args)
        exitos.append('evaluacion_exitosa' not in resultado or resultado.get('evaluacion_exitosa', True))
    
    # Generar reporte si se especifica
    if args.reporte:
        exito = generar_reporte(args)
        exitos.append(exito)
    
    # Mostrar resumen final
    if all(exitos):
        logger.info("=== TODAS LAS OPERACIONES COMPLETADAS EXITOSAMENTE ===")
        sys.exit(0)
    else:
        logger.error("=== ALGUNAS OPERACIONES FALLARON ===")
        sys.exit(1)


if __name__ == "__main__":
    main()