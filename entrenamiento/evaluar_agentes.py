"""
Script para evaluar y comparar diferentes agentes de ajedrez.
"""

import os
import sys
import logging
import argparse
import time
import random
import pandas as pd
import chess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentes.agente_minimax import AgenteMinimax
from agentes.agente_mcts import AgenteMCTS
from agentes.agente_dqn import AgenteDQN
from agentes.agente_ppo import AgentePPO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluacion_agentes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EvaluadorAgentes:
    """
    Clase para evaluar y comparar diferentes agentes de ajedrez.
    """
    
    def __init__(self, tiempo_limite_movimiento: float = 5.0):
        """
        Inicializa el evaluador.
        
        Args:
            tiempo_limite_movimiento: Tiempo límite por movimiento en segundos
        """
        self.tiempo_limite = tiempo_limite_movimiento
        self.resultados = []
        self.estadisticas = defaultdict(lambda: defaultdict(int))
        
    def cargar_agentes(self, config_agentes: Dict) -> Dict:
        """
        Carga los agentes según la configuración.
        
        Args:
            config_agentes: Configuración de agentes
            
        Returns:
            dict: Diccionario de agentes cargados
        """
        agentes = {}
        
        logger.info("Cargando agentes...")
        
        # Cargar Minimax
        if 'minimax' in config_agentes:
            config = config_agentes['minimax']
            agentes['minimax'] = AgenteMinimax(
                profundidad_maxima=config.get('profundidad', 3),
                tiempo_limite=config.get('tiempo_limite', self.tiempo_limite)
            )
            logger.info(f"Minimax cargado: profundidad {config.get('profundidad', 3)}")
        
        # Cargar MCTS
        if 'mcts' in config_agentes:
            config = config_agentes['mcts']
            agentes['mcts'] = AgenteMCTS(
                num_simulaciones=config.get('simulaciones', 1000),
                tiempo_limite=config.get('tiempo_limite', self.tiempo_limite),
                c_param=config.get('c_param', 1.4)
            )
            logger.info(f"MCTS cargado: {config.get('simulaciones', 1000)} simulaciones")
        
        # Cargar DQN
        if 'dqn' in config_agentes:
            config = config_agentes['dqn']
            modelo_path = config.get('modelo_path')
            if modelo_path and os.path.exists(modelo_path):
                agentes['dqn'] = AgenteDQN(modelo_path=modelo_path)
                logger.info(f"DQN cargado desde {modelo_path}")
            else:
                logger.warning(f"Modelo DQN no encontrado: {modelo_path}")
        
        # Cargar PPO
        if 'ppo' in config_agentes:
            config = config_agentes['ppo']
            modelo_path = config.get('modelo_path')
            if modelo_path and os.path.exists(modelo_path):
                agentes['ppo'] = AgentePPO(modelo_path=modelo_path)
                logger.info(f"PPO cargado desde {modelo_path}")
            else:
                logger.warning(f"Modelo PPO no encontrado: {modelo_path}")
        
        logger.info(f"Agentes cargados: {list(agentes.keys())}")
        return agentes
    
    def jugar_partida(self, agente1, agente2, nombre1: str, nombre2: str, 
                     max_movimientos: int = 200) -> Dict:
        """
        Simula una partida entre dos agentes.
        
        Args:
            agente1: Primer agente (juega con blancas)
            agente2: Segundo agente (juega con negras)
            nombre1: Nombre del primer agente
            nombre2: Nombre del segundo agente
            max_movimientos: Máximo número de movimientos
            
        Returns:
            dict: Resultado de la partida
        """
        tablero = chess.Board()
        movimientos_realizados = 0
        tiempos_agente1 = []
        tiempos_agente2 = []
        movimientos_partida = []
        
        tiempo_inicio_partida = time.time()
        
        while not tablero.is_game_over() and movimientos_realizados < max_movimientos:
            try:
                if tablero.turn == chess.WHITE:
                    # Turno del agente 1 (blancas)
                    tiempo_inicio = time.time()
                    movimiento = agente1.seleccionar_movimiento(tablero)
                    tiempo_movimiento = time.time() - tiempo_inicio
                    tiempos_agente1.append(tiempo_movimiento)
                    agente_actual = nombre1
                else:
                    # Turno del agente 2 (negras)
                    tiempo_inicio = time.time()
                    movimiento = agente2.seleccionar_movimiento(tablero)
                    tiempo_movimiento = time.time() - tiempo_inicio
                    tiempos_agente2.append(tiempo_movimiento)
                    agente_actual = nombre2
                
                # Verificar si el movimiento es válido
                if movimiento is None or movimiento not in tablero.legal_moves:
                    logger.warning(f"Movimiento inválido de {agente_actual}: {movimiento}")
                    # Seleccionar movimiento aleatorio como fallback
                    movimientos_legales = list(tablero.legal_moves)
                    if movimientos_legales:
                        movimiento = random.choice(movimientos_legales)
                    else:
                        break
                
                # Ejecutar movimiento
                tablero.push(movimiento)
                movimientos_partida.append(str(movimiento))
                movimientos_realizados += 1
                
                # Verificar límite de tiempo
                if tiempo_movimiento > self.tiempo_limite * 2:
                    logger.warning(f"{agente_actual} excedió tiempo límite: {tiempo_movimiento:.2f}s")
                
            except Exception as e:
                logger.error(f"Error en movimiento de {agente_actual}: {e}")
                break
        
        tiempo_total_partida = time.time() - tiempo_inicio_partida
        
        # Determinar resultado
        resultado = self._determinar_resultado(tablero, movimientos_realizados, max_movimientos)
        
        # Compilar estadísticas
        stats_partida = {
            'agente_blancas': nombre1,
            'agente_negras': nombre2,
            'resultado': resultado['resultado'],
            'ganador': resultado['ganador'],
            'razon_fin': resultado['razon'],
            'movimientos_realizados': movimientos_realizados,
            'tiempo_total_partida': tiempo_total_partida,
            'tiempo_promedio_blancas': sum(tiempos_agente1) / len(tiempos_agente1) if tiempos_agente1 else 0,
            'tiempo_promedio_negras': sum(tiempos_agente2) / len(tiempos_agente2) if tiempos_agente2 else 0,
            'tiempo_max_blancas': max(tiempos_agente1) if tiempos_agente1 else 0,
            'tiempo_max_negras': max(tiempos_agente2) if tiempos_agente2 else 0,
            'fen_final': tablero.fen(),
            'pgn': ' '.join(movimientos_partida)
        }
        
        return stats_partida
    
    def _determinar_resultado(self, tablero: chess.Board, movimientos: int, max_movimientos: int) -> Dict:
        """
        Determina el resultado de la partida.
        
        Args:
            tablero: Estado final del tablero
            movimientos: Movimientos realizados
            max_movimientos: Máximo de movimientos permitidos
            
        Returns:
            dict: Información del resultado
        """
        if tablero.is_checkmate():
            ganador = 'negras' if tablero.turn else 'blancas'
            return {
                'resultado': 1 if ganador == 'blancas' else -1,
                'ganador': ganador,
                'razon': 'jaque_mate'
            }
        elif tablero.is_stalemate():
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'ahogado'
            }
        elif tablero.is_insufficient_material():
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'material_insuficiente'
            }
        elif tablero.is_fivefold_repetition():
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'repeticion'
            }
        elif tablero.is_fifty_moves():
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'regla_50_movimientos'
            }
        elif movimientos >= max_movimientos:
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'limite_movimientos'
            }
        else:
            return {
                'resultado': 0,
                'ganador': 'empate',
                'razon': 'incompleta'
            }
    
    def enfrentar_agentes(self, agente1, agente2, nombre1: str, nombre2: str, 
                         num_partidas: int = 10) -> List[Dict]:
        """
        Enfrenta dos agentes múltiples veces.
        
        Args:
            agente1: Primer agente
            agente2: Segundo agente
            nombre1: Nombre del primer agente
            nombre2: Nombre del segundo agente
            num_partidas: Número de partidas a jugar
            
        Returns:
            list: Lista de resultados de partidas
        """
        logger.info(f"Enfrentando {nombre1} vs {nombre2} ({num_partidas} partidas)")
        
        resultados_enfrentamiento = []
        
        for i in range(num_partidas):
            logger.info(f"Partida {i+1}/{num_partidas}")
            
            # Alternar colores: partidas impares agente1=blancas, pares agente1=negras
            if i % 2 == 0:
                resultado = self.jugar_partida(agente1, agente2, nombre1, nombre2)
            else:
                resultado = self.jugar_partida(agente2, agente1, nombre2, nombre1)
            
            resultado['numero_partida'] = i + 1
            resultado['enfrentamiento'] = f"{nombre1}_vs_{nombre2}"
            
            resultados_enfrentamiento.append(resultado)
            
            # Log del resultado
            logger.info(f"Resultado: {resultado['ganador']} ({resultado['razon']}) "
                       f"en {resultado['movimientos_realizados']} movimientos")
        
        return resultados_enfrentamiento
    
    def evaluar_todos_contra_todos(self, agentes: Dict, num_partidas: int = 10) -> pd.DataFrame:
        """
        Evalúa todos los agentes contra todos.
        
        Args:
            agentes: Diccionario de agentes
            num_partidas: Número de partidas por enfrentamiento
            
        Returns:
            DataFrame: Resultados completos
        """
        nombres_agentes = list(agentes.keys())
        logger.info(f"Evaluación todos contra todos: {nombres_agentes}")
        
        todos_los_resultados = []
        
        for i, nombre1 in enumerate(nombres_agentes):
            for j, nombre2 in enumerate(nombres_agentes):
                if i != j:  # No enfrentar agente contra sí mismo
                    agente1 = agentes[nombre1]
                    agente2 = agentes[nombre2]
                    
                    resultados = self.enfrentar_agentes(
                        agente1, agente2, nombre1, nombre2, num_partidas
                    )
                    
                    todos_los_resultados.extend(resultados)
        
        self.resultados = todos_los_resultados
        return pd.DataFrame(todos_los_resultados)
    
    def generar_estadisticas_resumen(self, df_resultados: pd.DataFrame) -> Dict:
        """
        Genera estadísticas resumen de los resultados.
        
        Args:
            df_resultados: DataFrame con todos los resultados
            
        Returns:
            dict: Estadísticas resumidas
        """
        stats = {}
        
        # Obtener lista única de agentes
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = list(agentes_blancas.union(agentes_negras))
        
        for agente in todos_agentes:
            # Filtrar partidas donde participó este agente
            partidas_como_blancas = df_resultados[df_resultados['agente_blancas'] == agente]
            partidas_como_negras = df_resultados[df_resultados['agente_negras'] == agente]
            
            # Contar victorias, empates y derrotas
            victorias_blancas = len(partidas_como_blancas[partidas_como_blancas['ganador'] == 'blancas'])
            empates_blancas = len(partidas_como_blancas[partidas_como_blancas['ganador'] == 'empate'])
            derrotas_blancas = len(partidas_como_blancas[partidas_como_blancas['ganador'] == 'negras'])
            
            victorias_negras = len(partidas_como_negras[partidas_como_negras['ganador'] == 'negras'])
            empates_negras = len(partidas_como_negras[partidas_como_negras['ganador'] == 'empate'])
            derrotas_negras = len(partidas_como_negras[partidas_como_negras['ganador'] == 'blancas'])
            
            total_victorias = victorias_blancas + victorias_negras
            total_empates = empates_blancas + empates_negras
            total_derrotas = derrotas_blancas + derrotas_negras
            total_partidas = total_victorias + total_empates + total_derrotas
            
            # Calcular porcentajes
            if total_partidas > 0:
                porcentaje_victorias = (total_victorias / total_partidas) * 100
                porcentaje_empates = (total_empates / total_partidas) * 100
                porcentaje_derrotas = (total_derrotas / total_partidas) * 100
            else:
                porcentaje_victorias = porcentaje_empates = porcentaje_derrotas = 0
            
            # Calcular tiempos promedio
            todas_partidas = pd.concat([partidas_como_blancas, partidas_como_negras])
            tiempo_promedio_blancas = partidas_como_blancas['tiempo_promedio_blancas'].mean()
            tiempo_promedio_negras = partidas_como_negras['tiempo_promedio_negras'].mean()
            tiempo_promedio_general = (
                (tiempo_promedio_blancas + tiempo_promedio_negras) / 2 
                if not pd.isna(tiempo_promedio_blancas) and not pd.isna(tiempo_promedio_negras)
                else tiempo_promedio_blancas if not pd.isna(tiempo_promedio_blancas)
                else tiempo_promedio_negras if not pd.isna(tiempo_promedio_negras)
                else 0
            )
            
            stats[agente] = {
                'total_partidas': total_partidas,
                'victorias': total_victorias,
                'empates': total_empates,
                'derrotas': total_derrotas,
                'porcentaje_victorias': porcentaje_victorias,
                'porcentaje_empates': porcentaje_empates,
                'porcentaje_derrotas': porcentaje_derrotas,
                'puntuacion': total_victorias + (total_empates * 0.5),  # Sistema de puntos estándar
                'tiempo_promedio_por_movimiento': tiempo_promedio_general,
                'partidas_como_blancas': len(partidas_como_blancas),
                'partidas_como_negras': len(partidas_como_negras)
            }
        
        return stats
    
    def guardar_resultados(self, df_resultados: pd.DataFrame, stats_resumen: Dict, 
                          archivo_base: str = 'resultados/comparacion_agentes'):
        """
        Guarda los resultados en archivos.
        
        Args:
            df_resultados: DataFrame con resultados detallados
            stats_resumen: Estadísticas resumidas
            archivo_base: Ruta base para los archivos
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(archivo_base), exist_ok=True)
        
        # Guardar resultados detallados
        archivo_csv = f"{archivo_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_resultados.to_csv(archivo_csv, index=False)
        logger.info(f"Resultados detallados guardados en {archivo_csv}")
        
        # Guardar estadísticas resumen
        archivo_resumen = f"{archivo_base}_resumen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_resumen = pd.DataFrame.from_dict(stats_resumen, orient='index')
        df_resumen.to_csv(archivo_resumen)
        logger.info(f"Resumen guardado en {archivo_resumen}")
        
        # Guardar archivo de texto legible
        archivo_txt = f"{archivo_base}_reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(archivo_txt, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE EVALUACIÓN DE AGENTES DE AJEDREZ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("RESUMEN POR AGENTE:\n")
            f.write("-" * 30 + "\n")
            
            # Ordenar por puntuación
            agentes_ordenados = sorted(stats_resumen.items(), 
                                     key=lambda x: x[1]['puntuacion'], reverse=True)
            
            for i, (agente, stats) in enumerate(agentes_ordenados, 1):
                f.write(f"{i}. {agente.upper()}\n")
                f.write(f"   Partidas jugadas: {stats['total_partidas']}\n")
                f.write(f"   Victorias: {stats['victorias']} ({stats['porcentaje_victorias']:.1f}%)\n")
                f.write(f"   Empates: {stats['empates']} ({stats['porcentaje_empates']:.1f}%)\n")
                f.write(f"   Derrotas: {stats['derrotas']} ({stats['porcentaje_derrotas']:.1f}%)\n")
                f.write(f"   Puntuación: {stats['puntuacion']:.1f}\n")
                f.write(f"   Tiempo promedio: {stats['tiempo_promedio_por_movimiento']:.3f}s\n\n")
        
        logger.info(f"Reporte guardado en {archivo_txt}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Evaluar agentes de ajedrez')
    parser.add_argument('--partidas', type=int, default=10,
                       help='Número de partidas por enfrentamiento')
    parser.add_argument('--tiempo_limite', type=float, default=5.0,
                       help='Tiempo límite por movimiento en segundos')
    parser.add_argument('--output', type=str, default='resultados/comparacion_agentes',
                       help='Archivo base para guardar resultados')
    
    # Configuración de agentes
    parser.add_argument('--minimax_profundidad', type=int, default=3,
                       help='Profundidad para Minimax')
    parser.add_argument('--mcts_simulaciones', type=int, default=1000,
                       help='Simulaciones para MCTS')
    parser.add_argument('--dqn_modelo', type=str, default='modelos/dqn_modelo.zip',
                       help='Ruta del modelo DQN')
    parser.add_argument('--ppo_modelo', type=str, default='modelos/ppo_modelo.zip',
                       help='Ruta del modelo PPO')
    
    args = parser.parse_args()
    
    # Configurar agentes
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
    
    # Agregar agentes RL si los modelos existen
    if os.path.exists(args.dqn_modelo):
        config_agentes['dqn'] = {'modelo_path': args.dqn_modelo}
    else:
        logger.warning(f"Modelo DQN no encontrado: {args.dqn_modelo}")
    
    if os.path.exists(args.ppo_modelo):
        config_agentes['ppo'] = {'modelo_path': args.ppo_modelo}
    else:
        logger.warning(f"Modelo PPO no encontrado: {args.ppo_modelo}")
    
    # Crear evaluador
    evaluador = EvaluadorAgentes(tiempo_limite_movimiento=args.tiempo_limite)
    
    # Cargar agentes
    agentes = evaluador.cargar_agentes(config_agentes)
    
    if len(agentes) < 2:
        logger.error("Se necesitan al menos 2 agentes para la evaluación")
        return
    
    # Ejecutar evaluación
    logger.info("Iniciando evaluación completa...")
    tiempo_inicio = time.time()
    
    df_resultados = evaluador.evaluar_todos_contra_todos(agentes, args.partidas)
    
    tiempo_total = time.time() - tiempo_inicio
    logger.info(f"Evaluación completada en {tiempo_total:.2f} segundos")
    
    # Generar estadísticas
    stats_resumen = evaluador.generar_estadisticas_resumen(df_resultados)
    
    # Guardar resultados
    evaluador.guardar_resultados(df_resultados, stats_resumen, args.output)
    
    # Mostrar resumen en consola
    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    
    agentes_ordenados = sorted(stats_resumen.items(), 
                              key=lambda x: x[1]['puntuacion'], reverse=True)
    
    for i, (agente, stats) in enumerate(agentes_ordenados, 1):
        print(f"{i}. {agente.upper()}")
        print(f"   Puntuación: {stats['puntuación']:.1f}/{stats['total_partidas']}")
        print(f"   V/E/D: {stats['victorias']}/{stats['empates']}/{stats['derrotas']}")
        print(f"   Tiempo promedio: {stats['tiempo_promedio_por_movimiento']:.3f}s")
        print()


if __name__ == "__main__":
    main()
