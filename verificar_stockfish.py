"""
Script para verificar la instalación de Stockfish.

Este script verifica que Stockfish esté correctamente instalado
y puede comunicarse con python-chess.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verificar_stockfish():
    """Verifica la instalación de Stockfish."""
    print("="*60)
    print("VERIFICACIÓN DE STOCKFISH")
    print("="*60)
    
    try:
        from agentes.agente_stockfish import AgenteStockfish
        import chess
        
        print("\n✓ Módulos importados correctamente")
        
        # Intentar crear agente
        print("\nIntentando inicializar Stockfish...")
        agente = AgenteStockfish(nivel_habilidad=10, tiempo_limite=0.5)
        
        print("✓ Stockfish inicializado correctamente")
        print(f"  - Nivel: {agente.nivel_habilidad}")
        print(f"  - Tiempo límite: {agente.tiempo_limite}s")
        print(f"  - Ruta ejecutable: {agente.ruta_ejecutable}")
        
        # Probar evaluación
        print("\nProbando evaluación de posición inicial...")
        tablero = chess.Board()
        evaluacion = agente.obtener_evaluacion(tablero, tiempo=0.1)
        
        print(f"✓ Evaluación completada: {evaluacion} centipeones")
        
        # Probar selección de movimiento
        print("\nProbando selección de movimiento...")
        movimientos = list(tablero.legal_moves)
        info = {
            'tablero': tablero,
            'turno': tablero.turn,
            'jugadas_legales': len(movimientos)
        }
        
        accion = agente.seleccionar_accion(None, list(range(len(movimientos))), info)
        movimiento = movimientos[accion]
        
        print(f"✓ Movimiento seleccionado: {movimiento}")
        
        # Cerrar agente
        agente.cerrar()
        
        print("\n" + "="*60)
        print("✅ VERIFICACIÓN EXITOSA")
        print("="*60)
        print("\nStockfish está correctamente instalado y funcionando.")
        print("Puedes ejecutar:")
        print("  python ejecutar_stockfish.py --blancas stockfish --negras minimax")
        
        return True
        
    except FileNotFoundError as e:
        print("\n❌ ERROR: Stockfish no encontrado")
        print("\nPor favor, instala Stockfish:")
        print("  - Windows: https://stockfishchess.org/download/")
        print("  - Linux: sudo apt-get install stockfish")
        print("  - macOS: brew install stockfish")
        print("\nO especifica la ruta al ejecutable:")
        print("  python ejecutar_stockfish.py --ruta-stockfish /ruta/a/stockfish")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Función principal."""
    exito = verificar_stockfish()
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    main()
