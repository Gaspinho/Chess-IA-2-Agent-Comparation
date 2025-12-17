"""
Script para ejecutar partidas con Stockfish vs otros agentes.

Este script permite configurar y ejecutar partidas entre Stockfish
y otros agentes del proyecto (Minimax, MCTS, AlphaZero).
"""

import sys
import logging
import argparse
from typing import Dict, Any

from entorno.entorno_ajedrez import EntornoAjedrez
from agentes.agente_stockfish import AgenteStockfish
from agentes.agente_minimax import AgenteMinimax
from agentes.agente_mcts import AgenteMCTS

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crear_agente(tipo: str, config: Dict[str, Any]):
    """
    Crea un agente según el tipo especificado.
    
    Args:
        tipo: Tipo de agente ('stockfish', 'minimax', 'mcts')
        config: Configuración del agente
        
    Returns:
        Agente creado
    """
    if tipo.lower() == 'stockfish':
        return AgenteStockfish(
            nivel_habilidad=config.get('nivel_habilidad', 20),
            profundidad=config.get('profundidad'),
            tiempo_limite=config.get('tiempo_limite', 1.0),
            threads=config.get('threads', 1),
            hash_size=config.get('hash_size', 128)
        )
    elif tipo.lower() == 'minimax':
        return AgenteMinimax(
            profundidad_maxima=config.get('profundidad', 3),
            tiempo_limite=config.get('tiempo_limite', 5.0)
        )
    elif tipo.lower() == 'mcts':
        return AgenteMCTS(
            num_simulaciones=config.get('simulaciones', 1000),
            tiempo_limite=config.get('tiempo_limite', 5.0)
        )
    else:
        raise ValueError(f"Tipo de agente no soportado: {tipo}")


def ejecutar_partida(agente_blancas, agente_negras, max_jugadas: int = 200):
    """
    Ejecuta una partida entre dos agentes.
    
    Args:
        agente_blancas: Agente que juega con blancas
        agente_negras: Agente que juega con negras
        max_jugadas: Número máximo de jugadas
        
    Returns:
        Resultado de la partida
    """
    logger.info("="*60)
    logger.info("Iniciando partida: %s vs %s", agente_blancas.nombre, agente_negras.nombre)
    logger.info("="*60)
    
    entorno = EntornoAjedrez()
    observacion = entorno.reiniciar()
    
    agentes = {
        "player_0": agente_blancas,  # Blancas
        "player_1": agente_negras    # Negras
    }
    
    jugada_num = 0
    
    while not entorno.terminado and jugada_num < max_jugadas:
        # Obtener agente actual
        agente_id = entorno.agente_actual
        agente = agentes[agente_id]
        
        # Obtener movimientos legales
        movimientos_legales = list(entorno.tablero.legal_moves)
        acciones_legales = list(range(len(movimientos_legales)))
        
        # Preparar info para el agente
        info = {
            'tablero': entorno.tablero.copy(),
            'turno': entorno.tablero.turn,
            'jugadas_legales': len(movimientos_legales)
        }
        
        # Seleccionar acción
        logger.info("\nJugada %d - Turno de %s (%s)", 
                   jugada_num + 1, agente.nombre, "Blancas" if agente_id == "player_0" else "Negras")
        
        accion = agente.seleccionar_accion(observacion, acciones_legales, info)
        movimiento = movimientos_legales[accion]
        
        logger.info("Movimiento: %s", movimiento)
        
        # Ejecutar movimiento
        observacion, recompensa, terminado, info = entorno.paso(accion)
        jugada_num += 1
        
        # Mostrar estado del tablero
        logger.info("\n%s", entorno.tablero)
    
    # Resultado final
    logger.info("\n" + "="*60)
    logger.info("PARTIDA FINALIZADA")
    logger.info("="*60)
    
    resultado = entorno.tablero.result()
    logger.info("Resultado: %s", resultado)
    logger.info("Total de jugadas: %d", jugada_num)
    
    # Estadísticas de los agentes
    logger.info("\nEstadísticas de %s:", agente_blancas.nombre)
    stats_blancas = agente_blancas.obtener_estadisticas()
    for key, value in stats_blancas.items():
        logger.info("  %s: %s", key, value)
    
    logger.info("\nEstadísticas de %s:", agente_negras.nombre)
    stats_negras = agente_negras.obtener_estadisticas()
    for key, value in stats_negras.items():
        logger.info("  %s: %s", key, value)
    
    # Cerrar agentes Stockfish si existen
    if isinstance(agente_blancas, AgenteStockfish):
        agente_blancas.cerrar()
    if isinstance(agente_negras, AgenteStockfish):
        agente_negras.cerrar()
    
    return {
        'resultado': resultado,
        'jugadas': jugada_num,
        'stats_blancas': stats_blancas,
        'stats_negras': stats_negras
    }


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Ejecutar partidas con Stockfish')
    parser.add_argument('--blancas', type=str, default='stockfish', 
                       help='Tipo de agente para blancas (stockfish, minimax, mcts)')
    parser.add_argument('--negras', type=str, default='minimax',
                       help='Tipo de agente para negras (stockfish, minimax, mcts)')
    parser.add_argument('--nivel-stockfish', type=int, default=10,
                       help='Nivel de Stockfish (0-20)')
    parser.add_argument('--tiempo-stockfish', type=float, default=0.5,
                       help='Tiempo límite para Stockfish (segundos)')
    parser.add_argument('--profundidad-minimax', type=int, default=3,
                       help='Profundidad de búsqueda para Minimax')
    parser.add_argument('--tiempo-minimax', type=float, default=3.0,
                       help='Tiempo límite para Minimax (segundos)')
    parser.add_argument('--simulaciones-mcts', type=int, default=800,
                       help='Número de simulaciones para MCTS')
    parser.add_argument('--tiempo-mcts', type=float, default=3.0,
                       help='Tiempo límite para MCTS (segundos)')
    parser.add_argument('--max-jugadas', type=int, default=200,
                       help='Número máximo de jugadas')
    parser.add_argument('--ruta-stockfish', type=str, default=None,
                       help='Ruta al ejecutable de Stockfish')
    
    args = parser.parse_args()
    
    # Configuraciones de agentes
    config_stockfish = {
        'nivel_habilidad': args.nivel_stockfish,
        'tiempo_limite': args.tiempo_stockfish,
        'threads': 1,
        'hash_size': 128
    }
    
    config_minimax = {
        'profundidad': args.profundidad_minimax,
        'tiempo_limite': args.tiempo_minimax
    }
    
    config_mcts = {
        'simulaciones': args.simulaciones_mcts,
        'tiempo_limite': args.tiempo_mcts
    }
    
    # Crear agentes
    try:
        if args.blancas.lower() == 'stockfish':
            if args.ruta_stockfish:
                agente_blancas = AgenteStockfish(ruta_ejecutable=args.ruta_stockfish, **config_stockfish)
            else:
                agente_blancas = AgenteStockfish(**config_stockfish)
        elif args.blancas.lower() == 'minimax':
            agente_blancas = crear_agente('minimax', config_minimax)
        elif args.blancas.lower() == 'mcts':
            agente_blancas = crear_agente('mcts', config_mcts)
        else:
            logger.error("Tipo de agente no válido para blancas: %s", args.blancas)
            return
        
        if args.negras.lower() == 'stockfish':
            if args.ruta_stockfish:
                agente_negras = AgenteStockfish(ruta_ejecutable=args.ruta_stockfish, **config_stockfish)
            else:
                agente_negras = AgenteStockfish(**config_stockfish)
        elif args.negras.lower() == 'minimax':
            agente_negras = crear_agente('minimax', config_minimax)
        elif args.negras.lower() == 'mcts':
            agente_negras = crear_agente('mcts', config_mcts)
        else:
            logger.error("Tipo de agente no válido para negras: %s", args.negras)
            return
        
        # Ejecutar partida
        resultado = ejecutar_partida(agente_blancas, agente_negras, args.max_jugadas)
        
        logger.info("\n✓ Partida completada exitosamente")
        
    except FileNotFoundError as e:
        logger.error("\n✗ Error: No se encontró el ejecutable de Stockfish")
        logger.error("Instala Stockfish:")
        logger.error("  - Windows: https://stockfishchess.org/download/")
        logger.error("  - Linux: sudo apt-get install stockfish")
        logger.error("  - macOS: brew install stockfish")
        logger.error("\nO especifica la ruta con --ruta-stockfish")
    except Exception as e:
        logger.error("Error al ejecutar partida: %s", str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
