"""
Script de ejemplo para usar el agente AlphaZero.

Este script muestra cómo inicializar y usar el agente AlphaZero
tanto con un modelo pre-entrenado como con uno aleatorio.
"""

import os
import sys
import numpy as np

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentes.agente_alphazero import AgenteAlphaZero
import chess


def ejemplo_uso_basico():
    """Ejemplo básico de uso de AlphaZero."""
    print("=" * 60)
    print("  EJEMPLO: USO BÁSICO DE ALPHAZERO")
    print("=" * 60)
    print()
    
    # Crear agente (sin modelo pre-entrenado)
    agente = AgenteAlphaZero(
        num_simulaciones=400,  # Menos simulaciones para demo rápida
        c_puct=1.5,
        temperatura=1.0
    )
    
    print(f"Agente creado: {agente.nombre}")
    print(f"Simulaciones por jugada: {agente.num_simulaciones}")
    print(f"Dispositivo: {agente.dispositivo}")
    print()
    
    # Crear un tablero de ejemplo
    tablero = chess.Board()
    print("Tablero inicial:")
    print(tablero)
    print()
    
    # Simular info del entorno
    class MockLegalMoves:
        def __init__(self, board):
            self.board = board
    
    info = {'legal_moves': MockLegalMoves(tablero)}
    
    # Obtener acción del agente
    print("Pensando mejor jugada...")
    acciones_legales = np.arange(len(list(tablero.legal_moves)))
    
    accion = agente.seleccionar_accion(
        observacion=np.zeros((8, 8, 119)),  # Simplificado
        acciones_legales=acciones_legales,
        info=info
    )
    
    print(f"Acción seleccionada: {accion}")
    
    # Mostrar estadísticas
    stats = agente.obtener_estadisticas()
    print("\nEstadísticas del agente:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def ejemplo_con_modelo_entrenado():
    """Ejemplo usando un modelo pre-entrenado."""
    print("\n" + "=" * 60)
    print("  EJEMPLO: USO CON MODELO PRE-ENTRENADO")
    print("=" * 60)
    print()
    
    modelo_path = "modelos/alphazero_final.pth"
    
    if os.path.exists(modelo_path):
        print(f"Cargando modelo desde: {modelo_path}")
        agente = AgenteAlphaZero(
            num_simulaciones=800,
            modelo_path=modelo_path
        )
        print("✓ Modelo cargado exitosamente")
    else:
        print(f"⚠ Modelo no encontrado en: {modelo_path}")
        print("Entrenar primero con: python principal.py --entrenar alphazero")
        return
    
    # Jugar algunas jugadas de ejemplo
    tablero = chess.Board()
    
    for jugada_num in range(3):
        print(f"\n--- Jugada {jugada_num + 1} ---")
        print(tablero)
        
        class MockLegalMoves:
            def __init__(self, board):
                self.board = board
        
        info = {'legal_moves': MockLegalMoves(tablero)}
        acciones_legales = np.arange(len(list(tablero.legal_moves)))
        
        accion = agente.seleccionar_accion(
            observacion=np.zeros((8, 8, 119)),
            acciones_legales=acciones_legales,
            info=info
        )
        
        movimientos = list(tablero.legal_moves)
        if accion < len(movimientos):
            movimiento = movimientos[accion]
            tablero.push(movimiento)
            print(f"AlphaZero jugó: {movimiento.uci()}")


def ejemplo_comparacion_temperatura():
    """Ejemplo mostrando el efecto de la temperatura."""
    print("\n" + "=" * 60)
    print("  EJEMPLO: EFECTO DE LA TEMPERATURA")
    print("=" * 60)
    print()
    
    tablero = chess.Board()
    
    class MockLegalMoves:
        def __init__(self, board):
            self.board = board
    
    info = {'legal_moves': MockLegalMoves(tablero)}
    acciones_legales = np.arange(len(list(tablero.legal_moves)))
    
    temperaturas = [0.0, 0.5, 1.0, 1.5]
    
    for temp in temperaturas:
        print(f"\n--- Temperatura: {temp} ---")
        
        agente = AgenteAlphaZero(
            num_simulaciones=200,
            temperatura=temp
        )
        
        # Obtener 3 jugadas para ver variabilidad
        jugadas = []
        for _ in range(3):
            accion = agente.seleccionar_accion(
                observacion=np.zeros((8, 8, 119)),
                acciones_legales=acciones_legales,
                info=info
            )
            movimientos = list(tablero.legal_moves)
            if accion < len(movimientos):
                jugadas.append(movimientos[accion].uci())
        
        print(f"Jugadas seleccionadas: {jugadas}")
        
        if temp == 0.0:
            print("  → Temperatura 0: Siempre elige la mejor (determinístico)")
        elif temp == 1.0:
            print("  → Temperatura 1: Balance exploración/explotación")
        else:
            print(f"  → Temperatura {temp}: {'Más exploración' if temp > 1.0 else 'Más explotación'}")


def ejemplo_guardar_cargar():
    """Ejemplo de guardar y cargar modelo."""
    print("\n" + "=" * 60)
    print("  EJEMPLO: GUARDAR Y CARGAR MODELO")
    print("=" * 60)
    print()
    
    # Crear directorio si no existe
    os.makedirs("modelos", exist_ok=True)
    
    # Crear agente
    print("Creando agente...")
    agente1 = AgenteAlphaZero(num_simulaciones=100)
    
    # Guardar modelo
    ruta_modelo = "modelos/alphazero_ejemplo.pth"
    print(f"Guardando modelo en: {ruta_modelo}")
    agente1.guardar_modelo(ruta_modelo)
    print("✓ Modelo guardado")
    
    # Cargar en nuevo agente
    print(f"\nCargando modelo en nuevo agente...")
    agente2 = AgenteAlphaZero(
        num_simulaciones=100,
        modelo_path=ruta_modelo
    )
    print("✓ Modelo cargado exitosamente")
    
    # Verificar que funciona
    tablero = chess.Board()
    
    class MockLegalMoves:
        def __init__(self, board):
            self.board = board
    
    info = {'legal_moves': MockLegalMoves(tablero)}
    acciones_legales = np.arange(len(list(tablero.legal_moves)))
    
    print("\nProbando agente cargado...")
    accion = agente2.seleccionar_accion(
        observacion=np.zeros((8, 8, 119)),
        acciones_legales=acciones_legales,
        info=info
    )
    print(f"Acción generada: {accion}")
    print("✓ Agente funciona correctamente")


def main():
    """Función principal."""
    print("\n")
    print("*" * 60)
    print("  EJEMPLOS DE USO - AGENTE ALPHAZERO")
    print("*" * 60)
    
    try:
        # Ejecutar ejemplos
        ejemplo_uso_basico()
        ejemplo_comparacion_temperatura()
        ejemplo_guardar_cargar()
        ejemplo_con_modelo_entrenado()
        
        print("\n" + "=" * 60)
        print("  TODOS LOS EJEMPLOS COMPLETADOS")
        print("=" * 60)
        print("\nPara entrenar tu propio modelo:")
        print("  python principal.py --entrenar alphazero --num-iteraciones 100")
        print()
        
    except KeyboardInterrupt:
        print("\n\nEjemplos interrumpidos por el usuario")
    except Exception as e:
        print(f"\n✗ Error durante los ejemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
