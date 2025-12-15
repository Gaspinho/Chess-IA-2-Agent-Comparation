import chess
import time
import logging
import traceback
from agentes.agente_negamax_avanzado import AgenteNegamaxAvanzado
from analisis.visualizador_chess import VisualizadorAjedrez

logging.basicConfig(level=logging.INFO)

def jugar_partida():
    viz = VisualizadorAjedrez()
    tablero = chess.Board()
    
    # Tiempo reducido a 0.5s para evitar que la ventana se congele mucho
    agente_blancas = AgenteNegamaxAvanzado(tiempo_limite=5.0)
    agente_negras = AgenteNegamaxAvanzado(tiempo_limite=5.0)
    
    viz.actualizar_info_agentes({
        'agente_blancas': f"{agente_blancas.nombre} (Blanco)",
        'agente_negras': f"{agente_negras.nombre} (Negro)"
    })

    running = True
    ultima_jugada = None

    print("\n--- INICIO DE PARTIDA ---")

    while running:
        # 1. Verificar si el juego terminó
        if tablero.is_game_over():
            print(f"\n¡JUEGO TERMINADO! Resultado: {tablero.result()}")
            print(f"Razón: {tablero.outcome().termination}")
            break

        # 2. Manejar eventos de la ventana
        if not viz.manejar_eventos():
            running = False
            break

        # 3. Renderizar
        viz.actualizar_tablero(tablero, ultima_jugada)
        viz.renderizar()

        # 4. Turno del Agente
        es_turno_blancas = tablero.turn == chess.WHITE
        agente_actual = agente_blancas if es_turno_blancas else agente_negras
        color_texto = "Blancas" if es_turno_blancas else "Negras"
        
        try:
            print(f"Pensando ({color_texto})...", end='', flush=True)
            
            acciones_legales = list(tablero.legal_moves)
            
            # IMPORTANTE: Usar .copy() para evitar "Tablero Sucio"
            info = {'board': tablero.copy(), 'fen': tablero.fen()}
            
            # Llamada al agente
            movimiento = agente_actual.seleccionar_accion(
                observacion=None, 
                acciones_legales=acciones_legales, 
                info=info
            )
            print(f" Jugada: {movimiento}")
            
            tablero.push(movimiento)
            ultima_jugada = movimiento
            
            # Actualizar stats
            viz.actualizar_estadisticas({
                'Profundidad': agente_actual.stats['profundidad_alcanzada'],
                'Nodos calc.': agente_actual.stats['nodos'],
                'Valoración': agente_actual.tt.get(chess.polyglot.zobrist_hash(tablero), [0,0])[1]
            })

        except Exception as e:
            print("\n\n!!! ERROR EN EL TURNO !!!")
            traceback.print_exc() # Imprime el error completo
            break

    # Bucle final para mantener la ventana abierta
    print("Ventana en espera. Cierra la ventana para salir.")
    viz.actualizar_tablero(tablero, ultima_jugada)
    viz.renderizar()
    
    while running:
        if not viz.manejar_eventos():
            running = False
        viz.reloj.tick(30)

    viz.cerrar()

if __name__ == "__main__":
    jugar_partida()