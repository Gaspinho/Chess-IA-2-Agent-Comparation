"""
Archivo principal para el análisis comparativo de agentes de ajedrez.
Permite entrenar, evaluar y analizar diferentes tipos de agentes.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Optional

# Agregar módulos locales al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
def configurar_logging(nivel: str = 'INFO'):
    """Configura el sistema de logging."""
    nivel_logging = getattr(logging, nivel.upper(), logging.INFO)
    
    logging.basicConfig(
        level=nivel_logging,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'chess_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


def entrenar_agente_dqn(args):
    """Entrena un agente DQN."""
    logger.info("=== INICIANDO ENTRENAMIENTO DQN ===")
    
    try:
        from agentes.agente_dqn import AgenteDQN, CallbackEntrenamiento
        
        # Crear directorio de modelos
        os.makedirs('modelos', exist_ok=True)
        
        # Configurar parámetros
        parametros = {
            'learning_rate': args.learning_rate,
            'buffer_size': args.buffer_size,
            'batch_size': args.batch_size,
            'gamma': args.gamma,
            'target_update_interval': args.target_update_interval,
            'exploration_fraction': args.exploration_fraction,
            'exploration_initial_eps': args.exploration_initial_eps,
            'exploration_final_eps': args.exploration_final_eps,
            'train_freq': args.train_freq
        }
        
        logger.info(f"Parámetros DQN: {parametros}")
        
        # Crear y entrenar agente
        agente = AgenteDQN(**parametros)
        callback = CallbackEntrenamiento(verbose=1)
        
        tiempo_inicio = time.time()
        agente.entrenar(total_timesteps=args.timesteps, callback=callback)
        tiempo_total = time.time() - tiempo_inicio
        
        # Guardar modelo
        modelo_path = f'modelos/dqn_modelo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        agente.guardar_modelo(modelo_path)
        
        logger.info(f"Entrenamiento DQN completado en {tiempo_total:.2f}s")
        logger.info(f"Modelo guardado: {modelo_path}")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento DQN: {e}")
        raise


def entrenar_agente_ppo(args):
    """Entrena un agente PPO."""
    logger.info("=== INICIANDO ENTRENAMIENTO PPO ===")
    
    try:
        from agentes.agente_ppo import AgentePPO, CallbackEntrenamientoPPO
        
        # Crear directorio de modelos
        os.makedirs('modelos', exist_ok=True)
        
        # Configurar parámetros
        parametros = {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
            'max_grad_norm': args.max_grad_norm
        }
        
        logger.info(f"Parámetros PPO: {parametros}")
        
        # Crear y entrenar agente
        agente = AgentePPO(**parametros)
        callback = CallbackEntrenamientoPPO(verbose=1)
        
        tiempo_inicio = time.time()
        agente.entrenar(total_timesteps=args.timesteps, callback=callback)
        tiempo_total = time.time() - tiempo_inicio
        
        # Guardar modelo
        modelo_path = f'modelos/ppo_modelo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        agente.guardar_modelo(modelo_path)
        
        logger.info(f"Entrenamiento PPO completado en {tiempo_total:.2f}s")
        logger.info(f"Modelo guardado: {modelo_path}")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento PPO: {e}")
        raise


def evaluar_agentes(args):
    """Evalúa todos los agentes disponibles."""
    logger.info("=== INICIANDO EVALUACIÓN DE AGENTES ===")
    
    try:
        from entrenamiento.evaluar_agentes import EvaluadorAgentes
        
        # Configuración de agentes
        config_agentes = {
            'minimax': {
                'profundidad': args.minimax_profundidad,
                'tiempo_limite': args.tiempo_limite
            },
            'mcts': {
                'simulaciones': args.mcts_simulaciones,
                'tiempo_limite': args.tiempo_limite
            }
        }
        
        # Agregar agentes RL si existen modelos
        if args.dqn_modelo and os.path.exists(args.dqn_modelo):
            config_agentes['dqn'] = {'modelo_path': args.dqn_modelo}
        elif os.path.exists('modelos/dqn_modelo.zip'):
            config_agentes['dqn'] = {'modelo_path': 'modelos/dqn_modelo.zip'}
        
        if args.ppo_modelo and os.path.exists(args.ppo_modelo):
            config_agentes['ppo'] = {'modelo_path': args.ppo_modelo}
        elif os.path.exists('modelos/ppo_modelo.zip'):
            config_agentes['ppo'] = {'modelo_path': 'modelos/ppo_modelo.zip'}
        
        logger.info(f"Configuración de agentes: {list(config_agentes.keys())}")
        
        # Crear evaluador
        evaluador = EvaluadorAgentes(tiempo_limite_movimiento=args.tiempo_limite)
        
        # Cargar agentes
        agentes = evaluador.cargar_agentes(config_agentes)
        
        if len(agentes) < 2:
            logger.error("Se necesitan al menos 2 agentes para la evaluación")
            return
        
        # Ejecutar evaluación
        logger.info(f"Evaluando {len(agentes)} agentes con {args.partidas} partidas por enfrentamiento")
        tiempo_inicio = time.time()
        
        df_resultados = evaluador.evaluar_todos_contra_todos(agentes, args.partidas)
        
        tiempo_total = time.time() - tiempo_inicio
        logger.info(f"Evaluación completada en {tiempo_total:.2f} segundos")
        
        # Generar estadísticas
        stats_resumen = evaluador.generar_estadisticas_resumen(df_resultados)
        
        # Guardar resultados
        evaluador.guardar_resultados(df_resultados, stats_resumen, args.output)
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN")
        print("="*60)
        
        agentes_ordenados = sorted(stats_resumen.items(), 
                                  key=lambda x: x[1]['puntuacion'], reverse=True)
        
        for i, (agente, stats) in enumerate(agentes_ordenados, 1):
            print(f"{i}. {agente.upper()}")
            print(f"   Puntuación: {stats['puntuacion']:.1f}/{stats['total_partidas']}")
            print(f"   V/E/D: {stats['victorias']}/{stats['empates']}/{stats['derrotas']}")
            print(f"   Tiempo promedio: {stats['tiempo_promedio_por_movimiento']:.3f}s")
            print()
        
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        raise


def generar_reporte(args):
    """Genera el reporte de análisis completo."""
    logger.info("=== GENERANDO REPORTE DE ANÁLISIS ===")
    
    try:
        import pandas as pd
        from analisis.metricas_desempeno import CalculadorMetricas
        from analisis.visualizacion import GeneradorVisualizaciones
        from analisis.generar_reporte import GeneradorReportes
        
        # Buscar el archivo de resultados más reciente
        archivo_resultados = args.resultados
        if not archivo_resultados:
            # Buscar el más reciente en el directorio resultados
            import glob
            archivos = glob.glob('resultados/comparacion_agentes_*.csv')
            if archivos:
                archivo_resultados = max(archivos, key=os.path.getctime)
                logger.info(f"Usando archivo de resultados: {archivo_resultados}")
            else:
                logger.error("No se encontraron archivos de resultados")
                return
        
        if not os.path.exists(archivo_resultados):
            logger.error(f"Archivo de resultados no encontrado: {archivo_resultados}")
            return
        
        # Cargar datos
        logger.info("Cargando datos de resultados...")
        df_resultados = pd.read_csv(archivo_resultados)
        
        # Calcular métricas
        logger.info("Calculando métricas...")
        calculador = CalculadorMetricas()
        metricas_basicas = calculador.calcular_metricas_basicas(df_resultados)
        metricas_avanzadas = calculador.calcular_metricas_avanzadas(df_resultados)
        matriz_enfrentamientos = calculador.calcular_matriz_enfrentamientos(df_resultados)
        ratings_elo = calculador.calcular_rating_elo(df_resultados)
        
        # Generar visualizaciones
        if args.graficos:
            logger.info("Generando visualizaciones...")
            viz = GeneradorVisualizaciones()
            
            # Crear directorio de gráficos
            os.makedirs('resultados/graficos', exist_ok=True)
            
            archivos_graficos = {}
            
            # Generar gráficos individuales
            archivos_graficos['resultados'] = 'resultados/graficos/resultados_por_agente.png'
            viz.grafico_resultados_por_agente(metricas_basicas, archivos_graficos['resultados'])
            
            archivos_graficos['tasas_rendimiento'] = 'resultados/graficos/tasas_rendimiento.png'
            viz.grafico_tasas_rendimiento(metricas_basicas, archivos_graficos['tasas_rendimiento'])
            
            archivos_graficos['tiempos'] = 'resultados/graficos/tiempos_promedio.png'
            viz.grafico_tiempos_promedio(metricas_basicas, archivos_graficos['tiempos'])
            
            archivos_graficos['matriz'] = 'resultados/graficos/matriz_enfrentamientos.png'
            viz.grafico_matriz_enfrentamientos(matriz_enfrentamientos, archivos_graficos['matriz'])
            
            archivos_graficos['elo'] = 'resultados/graficos/evolucion_elo.png'
            viz.grafico_evolucion_elo(df_resultados, ratings_elo, archivos_graficos['elo'])
            
            archivos_graficos['movimientos'] = 'resultados/graficos/distribucion_movimientos.png'
            viz.grafico_distribucion_movimientos(df_resultados, archivos_graficos['movimientos'])
            
            archivos_graficos['radar'] = 'resultados/graficos/radar_metricas.png'
            viz.grafico_radar_metricas(metricas_basicas, archivos_graficos['radar'])
            
            archivos_graficos['dashboard'] = 'resultados/graficos/dashboard_completo.png'
            viz.dashboard_completo(metricas_basicas, df_resultados, matriz_enfrentamientos, 
                                  ratings_elo, archivos_graficos['dashboard'])
            
            # Limpiar memoria
            viz.cerrar_todas_figuras()
            
            logger.info(f"Gráficos generados en resultados/graficos/")
        else:
            archivos_graficos = {}
        
        # Generar reporte
        logger.info("Generando reporte...")
        generador = GeneradorReportes()
        
        archivo_reporte = args.output_reporte
        if not archivo_reporte:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archivo_reporte = f'analisis/reporte_completo_{timestamp}.md'
        
        # Crear directorio
        os.makedirs(os.path.dirname(archivo_reporte), exist_ok=True)
        
        # Generar reporte completo
        reporte_contenido = generador.generar_reporte_completo(
            metricas_basicas=metricas_basicas,
            metricas_avanzadas=metricas_avanzadas,
            df_resultados=df_resultados,
            matriz_enfrentamientos=matriz_enfrentamientos,
            ratings_elo=ratings_elo,
            archivos_graficos=archivos_graficos,
            archivo_salida=archivo_reporte
        )
        
        # Generar también reporte rápido
        archivo_rapido = archivo_reporte.replace('.md', '_rapido.md')
        generador.generar_reporte_rapido(metricas_basicas, ratings_elo, archivo_rapido)
        
        logger.info(f"Reporte completo generado: {archivo_reporte}")
        logger.info(f"Reporte rápido generado: {archivo_rapido}")
        
        # Mostrar resumen en consola
        print("\n" + "="*60)
        print("REPORTE GENERADO EXITOSAMENTE")
        print("="*60)
        print(f"📄 Reporte completo: {archivo_reporte}")
        print(f"⚡ Reporte rápido: {archivo_rapido}")
        if args.graficos:
            print(f"📊 Gráficos: resultados/graficos/")
        print()
        
        # Mostrar top 3 agentes
        agentes_ordenados = sorted(ratings_elo.items(), key=lambda x: x[1], reverse=True)
        print("🏆 TOP 3 AGENTES:")
        for i, (agente, rating) in enumerate(agentes_ordenados[:3], 1):
            puntos = metricas_basicas[agente]['porcentaje_puntos']
            print(f"{i}. {agente.upper()} - ELO: {rating:.0f} ({puntos:.1f}% puntos)")
        
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Sistema de Análisis Comparativo de Agentes de Ajedrez',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar agente DQN
  python principal.py --entrenar dqn --timesteps 50000

  # Entrenar agente PPO  
  python principal.py --entrenar ppo --timesteps 100000

  # Evaluar todos los agentes
  python principal.py --evaluar --partidas 20

  # Generar reporte completo con gráficos
  python principal.py --reporte --graficos

  # Pipeline completo: evaluar y generar reporte
  python principal.py --evaluar --reporte --partidas 15 --graficos
        """
    )
    
    # Comandos principales
    parser.add_argument('--entrenar', choices=['dqn', 'ppo'], 
                       help='Entrenar un agente específico')
    parser.add_argument('--evaluar', action='store_true',
                       help='Evaluar todos los agentes disponibles')
    parser.add_argument('--reporte', action='store_true',
                       help='Generar reporte de análisis')
    
    # Parámetros de entrenamiento DQN
    dqn_group = parser.add_argument_group('Parámetros DQN')
    dqn_group.add_argument('--learning_rate', type=float, default=1e-4)
    dqn_group.add_argument('--buffer_size', type=int, default=50000)
    dqn_group.add_argument('--batch_size', type=int, default=64)
    dqn_group.add_argument('--gamma', type=float, default=0.99)
    dqn_group.add_argument('--target_update_interval', type=int, default=1000)
    dqn_group.add_argument('--exploration_fraction', type=float, default=0.1)
    dqn_group.add_argument('--exploration_initial_eps', type=float, default=1.0)
    dqn_group.add_argument('--exploration_final_eps', type=float, default=0.05)
    dqn_group.add_argument('--train_freq', type=int, default=4)
    
    # Parámetros de entrenamiento PPO
    ppo_group = parser.add_argument_group('Parámetros PPO')
    ppo_group.add_argument('--n_steps', type=int, default=2048)
    ppo_group.add_argument('--n_epochs', type=int, default=10)
    ppo_group.add_argument('--gae_lambda', type=float, default=0.95)
    ppo_group.add_argument('--clip_range', type=float, default=0.2)
    ppo_group.add_argument('--ent_coef', type=float, default=0.01)
    ppo_group.add_argument('--vf_coef', type=float, default=0.5)
    ppo_group.add_argument('--max_grad_norm', type=float, default=0.5)
    
    # Parámetros de evaluación
    eval_group = parser.add_argument_group('Parámetros de Evaluación')
    eval_group.add_argument('--partidas', type=int, default=10,
                           help='Número de partidas por enfrentamiento')
    eval_group.add_argument('--tiempo_limite', type=float, default=5.0,
                           help='Tiempo límite por movimiento en segundos')
    eval_group.add_argument('--minimax_profundidad', type=int, default=3,
                           help='Profundidad de búsqueda para Minimax')
    eval_group.add_argument('--mcts_simulaciones', type=int, default=1000,
                           help='Simulaciones para MCTS')
    eval_group.add_argument('--dqn_modelo', type=str,
                           help='Ruta del modelo DQN a usar')
    eval_group.add_argument('--ppo_modelo', type=str,
                           help='Ruta del modelo PPO a usar')
    
    # Parámetros generales
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Timesteps de entrenamiento')
    parser.add_argument('--output', type=str, default='resultados/comparacion_agentes',
                       help='Archivo base para resultados de evaluación')
    parser.add_argument('--resultados', type=str,
                       help='Archivo CSV con resultados para generar reporte')
    parser.add_argument('--output_reporte', type=str,
                       help='Archivo de salida para el reporte')
    parser.add_argument('--graficos', action='store_true',
                       help='Generar gráficos en el reporte')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Nivel de logging')
    
    args = parser.parse_args()
    
    # Configurar logging
    configurar_logging(args.log_level)
    
    # Validar argumentos
    if not any([args.entrenar, args.evaluar, args.reporte]):
        parser.print_help()
        print("\nError: Debe especificar al menos una acción (--entrenar, --evaluar, o --reporte)")
        return
    
    # Ejecutar acciones
    try:
        inicio_total = time.time()
        
        logger.info("="*60)
        logger.info("SISTEMA DE ANÁLISIS DE AGENTES DE AJEDREZ")
        logger.info("="*60)
        
        # Entrenamiento
        if args.entrenar:
            if args.entrenar == 'dqn':
                entrenar_agente_dqn(args)
            elif args.entrenar == 'ppo':
                entrenar_agente_ppo(args)
        
        # Evaluación
        if args.evaluar:
            evaluar_agentes(args)
        
        # Reporte
        if args.reporte:
            generar_reporte(args)
        
        tiempo_total = time.time() - inicio_total
        
        logger.info("="*60)
        logger.info(f"PROCESO COMPLETADO EN {tiempo_total:.2f} SEGUNDOS")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        raise


if __name__ == "__main__":
    main()
