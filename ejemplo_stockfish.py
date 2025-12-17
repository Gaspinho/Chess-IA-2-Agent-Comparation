"""
Ejemplo simple de uso del agente Stockfish.

Este script muestra cómo usar el agente Stockfish de manera programática.
"""

import chess
from agentes.agente_stockfish import AgenteStockfish


def ejemplo_basico():
    """Ejemplo básico de evaluación con Stockfish."""
    print("="*60)
    print("EJEMPLO BÁSICO: EVALUACIÓN DE POSICIONES")
    print("="*60)
    
    # Crear agente Stockfish con nivel medio
    agente = AgenteStockfish(nivel_habilidad=10, tiempo_limite=0.5)
    
    # Evaluar posición inicial
    tablero = chess.Board()
    print("\nPosición inicial:")
    print(tablero)
    print()
    
    evaluacion = agente.obtener_evaluacion(tablero, tiempo=0.2)
    print(f"Evaluación: {evaluacion} centipeones")
    print("(0 = posición equilibrada)")
    
    # Hacer algunos movimientos y evaluar
    print("\n" + "-"*60)
    print("Después de 1.e4:")
    tablero.push_san("e4")
    print(tablero)
    print()
    
    evaluacion = agente.obtener_evaluacion(tablero, tiempo=0.2)
    print(f"Evaluación: {evaluacion} centipeones")
    
    # Otra jugada
    print("\n" + "-"*60)
    print("Después de 1.e4 e5:")
    tablero.push_san("e5")
    print(tablero)
    print()
    
    evaluacion = agente.obtener_evaluacion(tablero, tiempo=0.2)
    print(f"Evaluación: {evaluacion} centipeones")
    
    agente.cerrar()
    print("\n✓ Ejemplo completado")


def ejemplo_sugerencia_movimiento():
    """Ejemplo de obtener mejor movimiento."""
    print("\n" + "="*60)
    print("EJEMPLO: MEJOR MOVIMIENTO")
    print("="*60)
    
    agente = AgenteStockfish(nivel_habilidad=15, tiempo_limite=1.0)
    
    tablero = chess.Board()
    print("\nPosición inicial:")
    print(tablero)
    
    # Obtener mejor movimiento
    movimientos = list(tablero.legal_moves)
    info = {
        'tablero': tablero,
        'turno': tablero.turn,
        'jugadas_legales': len(movimientos)
    }
    
    print("\nStockfish está pensando...")
    accion = agente.seleccionar_accion(None, list(range(len(movimientos))), info)
    mejor_movimiento = movimientos[accion]
    
    print(f"Mejor movimiento según Stockfish: {mejor_movimiento}")
    
    # Mostrar estadísticas
    stats = agente.obtener_estadisticas()
    print(f"\nEstadísticas:")
    print(f"  - Jugadas realizadas: {stats['jugadas_realizadas']}")
    print(f"  - Tiempo total: {stats['tiempo_total']:.3f}s")
    if stats.get('evaluaciones'):
        print(f"  - Última evaluación: {stats['evaluaciones'][-1]}")
    
    agente.cerrar()
    print("\n✓ Ejemplo completado")


def ejemplo_comparacion_niveles():
    """Ejemplo comparando diferentes niveles de Stockfish."""
    print("\n" + "="*60)
    print("EJEMPLO: COMPARACIÓN DE NIVELES")
    print("="*60)
    
    tablero = chess.Board()
    tablero.push_san("e4")
    tablero.push_san("e5")
    tablero.push_san("Nf3")
    
    print("\nPosición:")
    print(tablero)
    print("\n" + str(tablero.fen()))
    
    niveles = [5, 10, 15, 20]
    movimientos = list(tablero.legal_moves)
    
    print("\nMovimiento sugerido por cada nivel:")
    print("-" * 40)
    
    for nivel in niveles:
        agente = AgenteStockfish(nivel_habilidad=nivel, tiempo_limite=0.3)
        
        info = {
            'tablero': tablero.copy(),
            'turno': tablero.turn,
            'jugadas_legales': len(movimientos)
        }
        
        accion = agente.seleccionar_accion(None, list(range(len(movimientos))), info)
        movimiento = movimientos[accion]
        
        print(f"Nivel {nivel:2d}: {movimiento}")
        
        agente.cerrar()
    
    print("\n✓ Ejemplo completado")


def main():
    """Función principal."""
    try:
        print("\n🏁 EJEMPLOS DE USO DE STOCKFISH\n")
        
        ejemplo_basico()
        ejemplo_sugerencia_movimiento()
        ejemplo_comparacion_niveles()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print("="*60)
        print("\nPara más información, ver:")
        print("  - STOCKFISH_GUIA.md")
        print("  - ejecutar_stockfish.py")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Stockfish no encontrado")
        print("\nPor favor, instala Stockfish:")
        print("  - Windows: https://stockfishchess.org/download/")
        print("  - Linux: sudo apt-get install stockfish")
        print("  - macOS: brew install stockfish")
        print("\nVer STOCKFISH_GUIA.md para más información")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
