"""
Script para ver a AlphaZero jugar contra otros agentes con visualización gráfica.

Este script permite ejecutar partidas entre AlphaZero y otros agentes
(Minimax, MCTS o AlphaZero vs AlphaZero) mostrando el juego en tiempo real.
"""

import logging
import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from entorno.entorno_ajedrez import EntornoAjedrez
from agentes.agente_minimax import AgenteMinimax
from agentes.agente_mcts import AgenteMCTS
from agentes.agente_alphazero import AgenteAlphaZero
from analisis.visualizador_chess import VisualizadorAjedrez
import chess
import pygame

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfiguracionPartida:
    """Configuración para una partida entre agentes."""
    agente_blancas: str
    agente_negras: str
    config_blancas: Dict[str, Any]
    config_negras: Dict[str, Any]
    velocidad_ms: int = 1000
    mostrar_estadisticas: bool = True


class VisualizadorAlphaZero:
    """
    Visualizador para partidas con AlphaZero.
    """
    
    def __init__(self, timeout_jugada: float = 10.0, max_jugadas: int = 500):
        """
        Inicializa el visualizador.
        
        Args:
            timeout_jugada: Tiempo máximo por jugada en segundos
            max_jugadas: Máximo número de jugadas por partida
        """
        self.timeout_jugada = timeout_jugada
        self.max_jugadas = max_jugadas
        self.visualizador = None
        
        logger.info("Visualizador AlphaZero inicializado")
    
    def crear_agentes(self, config: ConfiguracionPartida) -> tuple:
        """
        Crea los agentes según la configuración.
        
        Args:
            config: Configuración de la partida
            
        Returns:
            Tupla (agente_blancas, agente_negras)
        """
        agentes_disponibles = {
            'minimax': lambda cfg: AgenteMinimax(
                profundidad_maxima=cfg.get('profundidad', 3),
                tiempo_limite=cfg.get('tiempo_limite', 5.0)
            ),
            'mcts': lambda cfg: AgenteMCTS(
                num_simulaciones=cfg.get('simulaciones', 1000),
                tiempo_limite=cfg.get('tiempo_limite', 5.0),
                c_exploracion=cfg.get('c_exploracion', 1.414)
            ),
            'alphazero': lambda cfg: AgenteAlphaZero(
                num_simulaciones=cfg.get('simulaciones', 800),
                c_puct=cfg.get('c_puct', 1.5),
                temperatura=cfg.get('temperatura', 0.5),
                modelo_path=cfg.get('modelo_path', None)
            )
        }
        
        agente_blancas = agentes_disponibles[config.agente_blancas.lower()](config.config_blancas)
        agente_negras = agentes_disponibles[config.agente_negras.lower()](config.config_negras)
        
        logger.info("Agentes creados: %s (Blancas) vs %s (Negras)", 
                   agente_blancas.nombre, agente_negras.nombre)
        
        return agente_blancas, agente_negras
    
    def ejecutar_partida(self, config: ConfiguracionPartida) -> Dict[str, Any]:
        """
        Ejecuta una partida completa con visualización.
        
        Args:
            config: Configuración de la partida
            
        Returns:
            Diccionario con resultados de la partida
        """
        logger.info("=== INICIANDO PARTIDA VISUAL ===")
        logger.info("Blancas: %s", config.agente_blancas)
        logger.info("Negras: %s", config.agente_negras)
        
        # Crear agentes
        agente_blancas, agente_negras = self.crear_agentes(config)
        
        # Crear visualizador
        self.visualizador = VisualizadorAjedrez(tamaño_casilla=64)
        
        # Actualizar info de agentes en visualizador
        self.visualizador.info_agentes = {
            'blancas': agente_blancas.nombre,
            'negras': agente_negras.nombre
        }
        
        # Crear tablero
        tablero = chess.Board()
        
        # Variables de juego
        jugadas_realizadas = 0
        tiempo_blancas = 0.0
        tiempo_negras = 0.0
        historial_jugadas = []
        pausado = False
        
        # Loop principal
        corriendo = True
        reloj = pygame.time.Clock()
        
        while corriendo and not tablero.is_game_over() and jugadas_realizadas < self.max_jugadas:
            # Procesar eventos
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    corriendo = False
                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        corriendo = False
                    elif evento.key == pygame.K_SPACE:
                        pausado = not pausado
                        logger.info("Juego %s", "pausado" if pausado else "reanudado")
            
            if pausado:
                self.visualizador.tablero = tablero
                self.visualizador.dibujar_todo()
                pygame.display.flip()
                reloj.tick(30)
                continue
            
            # Determinar agente actual
            if tablero.turn == chess.WHITE:
                agente_actual = agente_blancas
                es_blancas = True
            else:
                agente_actual = agente_negras
                es_blancas = False
            
            logger.info("\n--- Jugada %d: %s ---", 
                       jugadas_realizadas + 1, 
                       "Blancas" if es_blancas else "Negras")
            
            # Obtener movimiento
            inicio = time.time()
            
            # Crear info simulada para el agente
            class MockLegalMoves:
                def __init__(self, board):
                    self.board = board
            
            info = {'legal_moves': MockLegalMoves(tablero)}
            movimientos_legales = list(tablero.legal_moves)
            
            if len(movimientos_legales) == 0:
                logger.warning("No hay movimientos legales")
                break
            
            # Seleccionar acción
            import numpy as np
            acciones_legales = np.arange(len(movimientos_legales))
            
            try:
                accion_idx = agente_actual.seleccionar_accion(
                    observacion=np.zeros((8, 8, 119)),
                    acciones_legales=acciones_legales,
                    info=info
                )
                
                movimiento = movimientos_legales[accion_idx]
                tiempo_jugada = time.time() - inicio
                
                # Actualizar tiempos
                if es_blancas:
                    tiempo_blancas += tiempo_jugada
                else:
                    tiempo_negras += tiempo_jugada
                
                # Realizar movimiento
                tablero.push(movimiento)
                historial_jugadas.append(movimiento)
                jugadas_realizadas += 1
                
                logger.info("Movimiento: %s (%.2fs)", movimiento.uci(), tiempo_jugada)
                
                # Actualizar visualizador
                self.visualizador.tablero = tablero
                self.visualizador.ultima_jugada = movimiento
                self.visualizador.estadisticas_partida = {
                    'jugadas': jugadas_realizadas,
                    'tiempo_blancas': tiempo_blancas,
                    'tiempo_negras': tiempo_negras,
                    'ultima_jugada': movimiento.uci(),
                    'tiempo_ultima': tiempo_jugada
                }
                
                # Dibujar
                self.visualizador.dibujar_todo()
                pygame.display.flip()
                
                # Esperar según velocidad configurada
                pygame.time.delay(config.velocidad_ms)
                
            except Exception as e:
                logger.error("Error al obtener movimiento: %s", str(e))
                break
        
        # Resultados finales
        resultado = self._analizar_resultado(tablero)
        resultado['jugadas_totales'] = jugadas_realizadas
        resultado['tiempo_blancas'] = tiempo_blancas
        resultado['tiempo_negras'] = tiempo_negras
        resultado['historial'] = historial_jugadas
        
        # Mostrar resultado en pantalla
        self._mostrar_resultado_final(resultado, agente_blancas.nombre, agente_negras.nombre)
        
        # Esperar antes de cerrar
        logger.info("\nPresiona cualquier tecla para cerrar...")
        esperando = True
        while esperando:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT or evento.type == pygame.KEYDOWN:
                    esperando = False
            pygame.time.delay(100)
        
        pygame.quit()
        
        return resultado
    
    def _analizar_resultado(self, tablero: chess.Board) -> Dict[str, Any]:
        """Analiza el resultado final de la partida."""
        resultado = {
            'terminado': tablero.is_game_over(),
            'ganador': None,
            'razon': None
        }
        
        if tablero.is_checkmate():
            resultado['ganador'] = 'Negras' if tablero.turn == chess.WHITE else 'Blancas'
            resultado['razon'] = 'Jaque mate'
        elif tablero.is_stalemate():
            resultado['razon'] = 'Ahogado (empate)'
        elif tablero.is_insufficient_material():
            resultado['razon'] = 'Material insuficiente (empate)'
        elif tablero.is_seventyfive_moves():
            resultado['razon'] = 'Regla de 75 movimientos (empate)'
        elif tablero.is_fivefold_repetition():
            resultado['razon'] = 'Repetición quíntuple (empate)'
        elif tablero.can_claim_draw():
            resultado['razon'] = 'Empate reclamable'
        
        return resultado
    
    def _mostrar_resultado_final(self, resultado: Dict[str, Any], 
                                 nombre_blancas: str, nombre_negras: str):
        """Muestra el resultado final en pantalla y consola."""
        logger.info("\n" + "=" * 60)
        logger.info("RESULTADO FINAL")
        logger.info("=" * 60)
        
        if resultado['ganador']:
            logger.info("🏆 GANADOR: %s", resultado['ganador'])
            if resultado['ganador'] == 'Blancas':
                logger.info("   Agente: %s", nombre_blancas)
            else:
                logger.info("   Agente: %s", nombre_negras)
        else:
            logger.info("🤝 EMPATE")
        
        logger.info("Razón: %s", resultado['razon'])
        logger.info("Jugadas totales: %d", resultado['jugadas_totales'])
        logger.info("Tiempo Blancas (%s): %.2fs", nombre_blancas, resultado['tiempo_blancas'])
        logger.info("Tiempo Negras (%s): %.2fs", nombre_negras, resultado['tiempo_negras'])
        logger.info("=" * 60)


def obtener_configuraciones_predefinidas() -> Dict[str, ConfiguracionPartida]:
    """Retorna las configuraciones predefinidas."""
    return {
        'alphazero_vs_minimax_rapido': ConfiguracionPartida(
            agente_blancas='alphazero',
            agente_negras='minimax',
            config_blancas={'simulaciones': 400, 'temperatura': 0.5},
            config_negras={'profundidad': 2, 'tiempo_limite': 3.0},
            velocidad_ms=800
        ),
        'alphazero_vs_minimax_normal': ConfiguracionPartida(
            agente_blancas='alphazero',
            agente_negras='minimax',
            config_blancas={'simulaciones': 800, 'temperatura': 0.5},
            config_negras={'profundidad': 3, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        ),
        'alphazero_vs_mcts_rapido': ConfiguracionPartida(
            agente_blancas='alphazero',
            agente_negras='mcts',
            config_blancas={'simulaciones': 400, 'temperatura': 0.5},
            config_negras={'simulaciones': 500, 'tiempo_limite': 3.0},
            velocidad_ms=800
        ),
        'alphazero_vs_mcts_normal': ConfiguracionPartida(
            agente_blancas='alphazero',
            agente_negras='mcts',
            config_blancas={'simulaciones': 800, 'temperatura': 0.5},
            config_negras={'simulaciones': 1000, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        ),
        'minimax_vs_alphazero': ConfiguracionPartida(
            agente_blancas='minimax',
            agente_negras='alphazero',
            config_blancas={'profundidad': 3, 'tiempo_limite': 5.0},
            config_negras={'simulaciones': 800, 'temperatura': 0.5},
            velocidad_ms=1000
        ),
        'mcts_vs_alphazero': ConfiguracionPartida(
            agente_blancas='mcts',
            agente_negras='alphazero',
            config_blancas={'simulaciones': 1000, 'tiempo_limite': 5.0},
            config_negras={'simulaciones': 800, 'temperatura': 0.5},
            velocidad_ms=1000
        ),
        'alphazero_vs_alphazero': ConfiguracionPartida(
            agente_blancas='alphazero',
            agente_negras='alphazero',
            config_blancas={'simulaciones': 800, 'temperatura': 0.8},
            config_negras={'simulaciones': 800, 'temperatura': 0.8},
            velocidad_ms=1000
        )
    }


def main():
    """Función principal."""
    print("=" * 60)
    print("    VER ALPHAZERO JUGAR - VISUALIZACIÓN GRÁFICA")
    print("=" * 60)
    print()
    
    # Verificar dependencias
    try:
        import pygame
        import chess
        import torch
        print("✓ Dependencias verificadas")
    except ImportError as e:
        print(f"✗ Error: Dependencia faltante - {e}")
        print("Instalar con: pip install pygame chess torch")
        return 1
    
    # Configuraciones disponibles
    configs = obtener_configuraciones_predefinidas()
    
    print("\nConfiguraciones disponibles:")
    print("1. AlphaZero vs Minimax (Rápido)")
    print("2. AlphaZero vs Minimax (Normal)")
    print("3. AlphaZero vs MCTS (Rápido)")
    print("4. AlphaZero vs MCTS (Normal)")
    print("5. Minimax vs AlphaZero (Normal)")
    print("6. MCTS vs AlphaZero (Normal)")
    print("7. AlphaZero vs AlphaZero (Auto-juego)")
    print()
    
    # Selección de configuración
    try:
        opcion = input("Selecciona una configuración (1-7) [2]: ").strip()
        if not opcion:
            opcion = "2"
        
        config_map = {
            "1": 'alphazero_vs_minimax_rapido',
            "2": 'alphazero_vs_minimax_normal',
            "3": 'alphazero_vs_mcts_rapido',
            "4": 'alphazero_vs_mcts_normal',
            "5": 'minimax_vs_alphazero',
            "6": 'mcts_vs_alphazero',
            "7": 'alphazero_vs_alphazero'
        }
        
        if opcion not in config_map:
            print("Opción inválida, usando configuración por defecto (2)")
            opcion = "2"
        
        config = configs[config_map[opcion]]
        
    except KeyboardInterrupt:
        print("\n\nCancelado por el usuario")
        return 0
    
    print()
    print("Configuración seleccionada:")
    print(f"  Blancas: {config.agente_blancas.upper()}")
    print(f"  Negras: {config.agente_negras.upper()}")
    print()
    print("Controles:")
    print("  ESC    - Salir")
    print("  SPACE  - Pausar/Reanudar")
    print()
    print("Iniciando partida...")
    print()
    
    # Ejecutar partida
    visualizador = VisualizadorAlphaZero()
    resultado = visualizador.ejecutar_partida(config)
    
    print("\n✓ Partida completada")
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
        exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
