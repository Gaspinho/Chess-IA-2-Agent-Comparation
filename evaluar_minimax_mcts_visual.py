"""
Evaluador específico para enfrentamientos Minimax vs MCTS con visualización gráfica.

Este script permite ejecutar partidas entre estos dos agentes específicos
mostrando el juego en tiempo real con una interfaz gráfica.
"""

import logging
import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entorno.entorno_ajedrez import EntornoAjedrez
from agentes.agente_minimax import AgenteMinimax
from agentes.agente_mcts import AgenteMCTS
from analisis.visualizador_chess import VisualizadorAjedrez
import chess

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfiguracionPartida:
    """Configuración para una partida entre agentes."""
    agente_blancas: str
    agente_negras: str
    config_minimax: Dict[str, Any]
    config_mcts: Dict[str, Any]
    velocidad_ms: int = 1000
    mostrar_estadisticas: bool = True


class EvaluadorVisualMinimaxMCTS:
    """
    Evaluador específico para Minimax vs MCTS con visualización gráfica.
    """
    
    def __init__(self, timeout_jugada: float = 10.0, max_jugadas: int = 500):
        """
        Inicializa el evaluador.
        
        Args:
            timeout_jugada: Tiempo máximo por jugada en segundos
            max_jugadas: Máximo número de jugadas por partida
        """
        self.timeout_jugada = timeout_jugada
        self.max_jugadas = max_jugadas
        self.visualizador = None
        
        logger.info("Evaluador visual Minimax vs MCTS inicializado")
    
    def crear_agentes(self, config: ConfiguracionPartida) -> tuple:
        """
        Crea los agentes Minimax y MCTS según la configuración.
        
        Args:
            config: Configuración de la partida
            
        Returns:
            Tupla (agente_blancas, agente_negras)
        """
        # Crear agente Minimax
        minimax = AgenteMinimax(
            profundidad_maxima=config.config_minimax.get('profundidad', 3),
            tiempo_limite=config.config_minimax.get('tiempo_limite', 5.0)
        )
        
        # Crear agente MCTS
        mcts = AgenteMCTS(
            num_simulaciones=config.config_mcts.get('simulaciones', 1000),
            tiempo_limite=config.config_mcts.get('tiempo_limite', 5.0),
            c_exploracion=config.config_mcts.get('c_exploracion', 1.414)
        )
        
        # Asignar según configuración
        if config.agente_blancas.lower() == 'minimax':
            return minimax, mcts
        else:
            return mcts, minimax
    
    def convertir_chess_board_a_entorno(self, chess_board: chess.Board, entorno: EntornoAjedrez):
        """
        Sincroniza un tablero de chess con el entorno de PettingZoo.
        
        Args:
            chess_board: Tablero de la librería chess
            entorno: Entorno de PettingZoo
        """
        # Obtener el FEN del tablero chess
        fen = chess_board.fen()
        
        # Aplicar el FEN al entorno (esto puede requerir adaptación según la implementación)
        # Por ahora, aplicamos las jugadas una por una
        if hasattr(entorno.env, 'board'):
            entorno.env.board = chess_board.copy()
    
    def jugar_partida_visual(self, config: ConfiguracionPartida) -> Dict[str, Any]:
        """
        Juega una partida entre Minimax y MCTS con visualización.
        
        Args:
            config: Configuración de la partida
            
        Returns:
            Diccionario con el resultado de la partida
        """
        # Crear visualizador
        self.visualizador = VisualizadorAjedrez(tamaño_casilla=64)
        
        # Crear agentes
        agente_blancas, agente_negras = self.crear_agentes(config)
        
        # Configurar información inicial
        self.visualizador.actualizar_info_agentes({
            'agente_blancas': config.agente_blancas,
            'agente_negras': config.agente_negras
        })
        
        # Crear entorno
        entorno = EntornoAjedrez(render_mode=None)  # Sin render ya que usamos nuestro visualizador
        
        # Crear tablero chess para control directo
        tablero_chess = chess.Board()
        
        # Variables de la partida
        inicio_partida = time.time()
        tiempo_blancas = 0.0
        tiempo_negras = 0.0
        num_jugadas = 0
        ultima_jugada = None
        resultado = None
        
        logger.info("=== INICIANDO PARTIDA %s (BLANCAS) vs %s (NEGRAS) ===", 
                   config.agente_blancas, config.agente_negras)
        
        try:
            # Renderizado inicial
            self.visualizador.actualizar_tablero(tablero_chess)
            self.visualizador.actualizar_estadisticas({
                'Tiempo Blancas': f"{tiempo_blancas:.1f}s",
                'Tiempo Negras': f"{tiempo_negras:.1f}s",
                'Jugadas': num_jugadas
            })
            self.visualizador.renderizar()
            
            # Esperar inicio
            if not self.visualizador.esperar(1000):
                return self._crear_resultado_error("Usuario canceló")
            
            # Bucle principal de la partida
            while not tablero_chess.is_game_over() and num_jugadas < self.max_jugadas:
                # Verificar eventos
                if not self.visualizador.manejar_eventos():
                    return self._crear_resultado_error("Usuario canceló")
                
                # Determinar agente actual
                if tablero_chess.turn:  # Blancas
                    agente_actual = agente_blancas
                    es_blancas = True
                    nombre_agente = config.agente_blancas
                else:  # Negras
                    agente_actual = agente_negras
                    es_blancas = False
                    nombre_agente = config.agente_negras
                
                logger.info("Turno %d: %s (%s)", num_jugadas + 1, 
                           nombre_agente, "Blancas" if es_blancas else "Negras")
                
                # Obtener acciones legales
                movimientos_legales = list(tablero_chess.legal_moves)
                
                if len(movimientos_legales) == 0:
                    break
                
                # Convertir movimientos a formato del agente
                acciones_legales = list(range(len(movimientos_legales)))
                
                # Ejecutar jugada con timeout
                inicio_jugada = time.time()
                try:
                    # Crear observación simulada (puedes mejorar esto)
                    observacion = self._crear_observacion_desde_tablero(tablero_chess)
                    info = {'legal_moves': movimientos_legales}
                    
                    # Seleccionar acción
                    if hasattr(agente_actual, 'seleccionar_accion'):
                        indice_accion = agente_actual.seleccionar_accion(
                            observacion, acciones_legales, info
                        )
                        jugada = movimientos_legales[indice_accion]
                    else:
                        # Método alternativo si el agente usa chess.Move directamente
                        jugada = agente_actual.obtener_mejor_jugada(tablero_chess)
                    
                    tiempo_jugada = time.time() - inicio_jugada
                    
                    # Verificar timeout
                    if tiempo_jugada > self.timeout_jugada:
                        logger.warning("Timeout en jugada %d para %s", num_jugadas, nombre_agente)
                        return self._crear_resultado_timeout(num_jugadas, tiempo_blancas, tiempo_negras)
                    
                    # Aplicar jugada
                    tablero_chess.push(jugada)
                    ultima_jugada = jugada
                    
                    # Acumular tiempo
                    if es_blancas:
                        tiempo_blancas += tiempo_jugada
                    else:
                        tiempo_negras += tiempo_jugada
                    
                    num_jugadas += 1
                    
                    logger.info("Jugada %d: %s (%.2fs)", num_jugadas, jugada.uci(), tiempo_jugada)
                    
                    # Actualizar visualización
                    self.visualizador.actualizar_tablero(tablero_chess, ultima_jugada)
                    self.visualizador.actualizar_estadisticas({
                        'Tiempo Blancas': f"{tiempo_blancas:.1f}s",
                        'Tiempo Negras': f"{tiempo_negras:.1f}s",
                        'Jugadas': num_jugadas,
                        'Última jugada': f"{jugada.uci()} ({tiempo_jugada:.2f}s)"
                    })
                    self.visualizador.renderizar()
                    
                    # Esperar según velocidad configurada
                    if not self.visualizador.esperar(config.velocidad_ms):
                        return self._crear_resultado_error("Usuario canceló")
                    
                except Exception as e:
                    logger.error("Error en jugada %d: %s", num_jugadas, str(e))
                    return self._crear_resultado_error(f"Error en jugada: {str(e)}")
            
            # Determinar resultado final
            tiempo_total = time.time() - inicio_partida
            
            if tablero_chess.is_checkmate():
                ganador = config.agente_blancas if not tablero_chess.turn else config.agente_negras
                tipo_fin = 'jaque_mate'
                logger.info("¡JAQUE MATE! Ganador: %s", ganador)
            elif tablero_chess.is_stalemate():
                ganador = None
                tipo_fin = 'empate_ahogado'
                logger.info("EMPATE por ahogado")
            elif tablero_chess.is_insufficient_material():
                ganador = None
                tipo_fin = 'empate_material'
                logger.info("EMPATE por material insuficiente")
            elif tablero_chess.is_seventyfive_moves():
                ganador = None
                tipo_fin = 'empate_75_jugadas'
                logger.info("EMPATE por regla de 75 jugadas")
            elif tablero_chess.is_fivefold_repetition():
                ganador = None
                tipo_fin = 'empate_repeticion'
                logger.info("EMPATE por repetición")
            elif num_jugadas >= self.max_jugadas:
                ganador = None
                tipo_fin = 'empate_limite_jugadas'
                logger.info("EMPATE por límite de jugadas")
            else:
                ganador = None
                tipo_fin = 'empate_otros'
                logger.info("EMPATE por otras razones")
            
            # Mostrar resultado final
            self.visualizador.actualizar_estadisticas({
                'Tiempo Blancas': f"{tiempo_blancas:.1f}s",
                'Tiempo Negras': f"{tiempo_negras:.1f}s",
                'Jugadas': num_jugadas,
                'Tiempo Total': f"{tiempo_total:.1f}s",
                'Resultado': tipo_fin.replace('_', ' ').title(),
                'Ganador': ganador or 'Empate'
            })
            self.visualizador.renderizar()
            
            logger.info("=== PARTIDA FINALIZADA ===")
            logger.info("Resultado: %s", tipo_fin)
            logger.info("Ganador: %s", ganador or "Empate")
            logger.info("Jugadas: %d", num_jugadas)
            logger.info("Tiempo total: %.1fs", tiempo_total)
            
            # Esperar antes de cerrar
            logger.info("Presiona ESC para cerrar o espera 10 segundos...")
            self.visualizador.esperar(10000)
            
            return {
                'ganador': ganador,
                'tipo_fin': tipo_fin,
                'num_jugadas': num_jugadas,
                'tiempo_total': tiempo_total,
                'tiempo_blancas': tiempo_blancas,
                'tiempo_negras': tiempo_negras,
                'agente_blancas': config.agente_blancas,
                'agente_negras': config.agente_negras,
                'exitoso': True
            }
            
        except Exception as e:
            logger.error("Error durante la partida: %s", str(e))
            return self._crear_resultado_error(f"Error durante partida: {str(e)}")
        
        finally:
            if self.visualizador:
                self.visualizador.cerrar()
    
    def _crear_observacion_desde_tablero(self, tablero: chess.Board):
        """Crea una observación simulada desde un tablero chess."""
        # Esto es una implementación simplificada
        # En una implementación completa, deberías convertir el tablero a la 
        # representación que esperan los agentes
        return tablero.fen().encode()
    
    def _crear_resultado_error(self, mensaje: str) -> Dict[str, Any]:
        """Crea un resultado de error."""
        return {
            'ganador': None,
            'tipo_fin': 'error',
            'error': mensaje,
            'exitoso': False
        }
    
    def _crear_resultado_timeout(self, num_jugadas: int, tiempo_blancas: float, 
                                tiempo_negras: float) -> Dict[str, Any]:
        """Crea un resultado de timeout."""
        return {
            'ganador': None,
            'tipo_fin': 'timeout',
            'num_jugadas': num_jugadas,
            'tiempo_blancas': tiempo_blancas,
            'tiempo_negras': tiempo_negras,
            'exitoso': False
        }


def crear_configuraciones_predefinidas() -> Dict[str, ConfiguracionPartida]:
    """Crea configuraciones predefinidas para diferentes tipos de partidas."""
    return {
        'minimax_vs_mcts_rapido': ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 2, 'tiempo_limite': 3.0},
            config_mcts={'simulaciones': 500, 'tiempo_limite': 3.0},
            velocidad_ms=800
        ),
        'minimax_vs_mcts_normal': ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 3, 'tiempo_limite': 5.0},
            config_mcts={'simulaciones': 1000, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        ),
        'minimax_vs_mcts_fuerte': ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 4, 'tiempo_limite': 8.0},
            config_mcts={'simulaciones': 2000, 'tiempo_limite': 8.0},
            velocidad_ms=1200
        ),
        'mcts_vs_minimax_normal': ConfiguracionPartida(
            agente_blancas='MCTS',
            agente_negras='Minimax',
            config_minimax={'profundidad': 3, 'tiempo_limite': 5.0},
            config_mcts={'simulaciones': 1000, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        )
    }


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Evaluador visual Minimax vs MCTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_normal
  python evaluar_minimax_mcts_visual.py --config minimax_vs_mcts_rapido --velocidad 500
  python evaluar_minimax_mcts_visual.py --blancas minimax --negras mcts --profundidad 3 --simulaciones 1000
        """
    )
    
    parser.add_argument('--config', choices=['minimax_vs_mcts_rapido', 'minimax_vs_mcts_normal', 
                                           'minimax_vs_mcts_fuerte', 'mcts_vs_minimax_normal'],
                       default='minimax_vs_mcts_normal',
                       help='Configuración predefinida a usar')
    
    parser.add_argument('--blancas', choices=['minimax', 'mcts'], 
                       help='Agente para piezas blancas')
    parser.add_argument('--negras', choices=['minimax', 'mcts'],
                       help='Agente para piezas negras')
    
    parser.add_argument('--profundidad', type=int, default=3,
                       help='Profundidad de búsqueda para Minimax')
    parser.add_argument('--simulaciones', type=int, default=1000,
                       help='Número de simulaciones para MCTS')
    parser.add_argument('--tiempo-limite', type=float, default=5.0,
                       help='Tiempo límite por jugada en segundos')
    
    parser.add_argument('--velocidad', type=int, default=1000,
                       help='Velocidad de visualización en milisegundos')
    
    parser.add_argument('--max-jugadas', type=int, default=500,
                       help='Máximo número de jugadas por partida')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Activar logging detallado')
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear evaluador
    evaluador = EvaluadorVisualMinimaxMCTS(
        timeout_jugada=args.tiempo_limite + 2.0,  # Un poco más de margen
        max_jugadas=args.max_jugadas
    )
    
    # Configurar partida
    if args.blancas and args.negras:
        # Configuración personalizada
        config = ConfiguracionPartida(
            agente_blancas=args.blancas.title(),
            agente_negras=args.negras.title(),
            config_minimax={'profundidad': args.profundidad, 'tiempo_limite': args.tiempo_limite},
            config_mcts={'simulaciones': args.simulaciones, 'tiempo_limite': args.tiempo_limite},
            velocidad_ms=args.velocidad
        )
    else:
        # Configuración predefinida
        configuraciones = crear_configuraciones_predefinidas()
        config = configuraciones[args.config]
        
        # Aplicar override de velocidad si se especifica
        if args.velocidad != 1000:
            config.velocidad_ms = args.velocidad
    
    logger.info("=== EVALUADOR VISUAL MINIMAX vs MCTS ===")
    logger.info("Configuración: %s (Blancas) vs %s (Negras)", 
               config.agente_blancas, config.agente_negras)
    logger.info("Velocidad: %d ms por jugada", config.velocidad_ms)
    
    try:
        # Verificar dependencias
        try:
            import pygame
            import chess
        except ImportError as e:
            logger.error("Dependencia faltante: %s", str(e))
            logger.error("Instalar con: pip install -r requerimientos.txt")
            sys.exit(1)
        
        # Ejecutar partida
        resultado = evaluador.jugar_partida_visual(config)
        
        if resultado.get('exitoso', False):
            print("\n=== RESULTADO FINAL ===")
            print(f"Ganador: {resultado.get('ganador', 'Empate')}")
            print(f"Tipo de fin: {resultado.get('tipo_fin', 'N/A')}")
            print(f"Jugadas: {resultado.get('num_jugadas', 0)}")
            print(f"Tiempo total: {resultado.get('tiempo_total', 0):.1f}s")
            print(f"Tiempo blancas: {resultado.get('tiempo_blancas', 0):.1f}s")
            print(f"Tiempo negras: {resultado.get('tiempo_negras', 0):.1f}s")
        else:
            print(f"\\nError en la partida: {resultado.get('error', 'Error desconocido')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Evaluación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error("Error durante la evaluación: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()