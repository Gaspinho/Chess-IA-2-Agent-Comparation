"""
Script simple para ejecutar evaluación visual Minimax vs MCTS.

Uso:
  python ejecutar_minimax_vs_mcts.py
"""

import os
import sys
import logging

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluar_minimax_mcts_visual import EvaluadorVisualMinimaxMCTS, ConfiguracionPartida

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Función principal."""
    print("=" * 60)
    print("    EVALUACIÓN VISUAL: MINIMAX vs MCTS")
    print("=" * 60)
    print()
    
    # Verificar dependencias
    try:
        import pygame
        import chess
        print("✓ Dependencias verificadas")
    except ImportError as e:
        print(f"✗ Error: Dependencia faltante - {e}")
        print("Instalar con: pip install pygame chess")
        return 1
    
    # Configuraciones disponibles
    print("Configuraciones disponibles:")
    print("1. Minimax vs MCTS (Rápido) - 2 profundidad, 500 simulaciones")
    print("2. Minimax vs MCTS (Normal) - 3 profundidad, 1000 simulaciones") 
    print("3. Minimax vs MCTS (Fuerte) - 4 profundidad, 2000 simulaciones")
    print("4. MCTS vs Minimax (Normal) - 3 profundidad, 1000 simulaciones")
    print()
    
    # Selección de configuración
    while True:
        try:
            opcion = input("Selecciona una configuración (1-4) [2]: ").strip()
            if not opcion:
                opcion = "2"
            
            opcion_num = int(opcion)
            if 1 <= opcion_num <= 4:
                break
            else:
                print("Por favor, selecciona una opción válida (1-4)")
        except ValueError:
            print("Por favor, ingresa un número válido")
    
    # Configurar según selección
    configuraciones = {
        1: ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 2, 'tiempo_limite': 3.0},
            config_mcts={'simulaciones': 500, 'tiempo_limite': 3.0},
            velocidad_ms=800
        ),
        2: ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 3, 'tiempo_limite': 5.0},
            config_mcts={'simulaciones': 1000, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        ),
        3: ConfiguracionPartida(
            agente_blancas='Minimax',
            agente_negras='MCTS',
            config_minimax={'profundidad': 4, 'tiempo_limite': 8.0},
            config_mcts={'simulaciones': 2000, 'tiempo_limite': 8.0},
            velocidad_ms=1200
        ),
        4: ConfiguracionPartida(
            agente_blancas='MCTS',
            agente_negras='Minimax',
            config_minimax={'profundidad': 3, 'tiempo_limite': 5.0},
            config_mcts={'simulaciones': 1000, 'tiempo_limite': 5.0},
            velocidad_ms=1000
        )
    }
    
    config = configuraciones[opcion_num]
    
    print(f"\\n✓ Configuración seleccionada:")
    print(f"  Blancas: {config.agente_blancas}")
    print(f"  Negras: {config.agente_negras}")
    if config.agente_blancas == 'Minimax' or config.agente_negras == 'Minimax':
        print(f"  Minimax - Profundidad: {config.config_minimax['profundidad']}")
    if config.agente_blancas == 'MCTS' or config.agente_negras == 'MCTS':
        print(f"  MCTS - Simulaciones: {config.config_mcts['simulaciones']}")
    print(f"  Velocidad: {config.velocidad_ms}ms por jugada")
    print()
    
    # Controles
    print("CONTROLES:")
    print("  ESC - Salir del juego")
    print("  SPACE - Pausar/Reanudar")
    print("  R - Reiniciar (no implementado)")
    print()
    
    input("Presiona ENTER para comenzar la partida...")
    
    # Crear evaluador y ejecutar partida
    try:
        evaluador = EvaluadorVisualMinimaxMCTS(
            timeout_jugada=max(config.config_minimax['tiempo_limite'], 
                             config.config_mcts['tiempo_limite']) + 2.0,
            max_jugadas=500
        )
        
        print("\\n🎮 Iniciando partida...")
        resultado = evaluador.jugar_partida_visual(config)
        
        # Mostrar resultado
        print("\\n" + "=" * 60)
        print("    RESULTADO FINAL")
        print("=" * 60)
        
        if resultado.get('exitoso', False):
            print(f"Ganador: {resultado.get('ganador', 'Empate')}")
            print(f"Tipo de fin: {resultado.get('tipo_fin', 'N/A').replace('_', ' ').title()}")
            print(f"Número de jugadas: {resultado.get('num_jugadas', 0)}")
            print(f"Tiempo total: {resultado.get('tiempo_total', 0):.1f}s")
            print(f"Tiempo {config.agente_blancas} (Blancas): {resultado.get('tiempo_blancas', 0):.1f}s")
            print(f"Tiempo {config.agente_negras} (Negras): {resultado.get('tiempo_negras', 0):.1f}s")
            
            # Análisis básico
            if resultado.get('ganador'):
                if resultado['ganador'] == config.agente_blancas:
                    print(f"\\n🏆 {config.agente_blancas} ganó jugando con blancas")
                else:
                    print(f"\\n🏆 {config.agente_negras} ganó jugando con negras")
            else:
                print("\\n🤝 La partida terminó en empate")
        else:
            print(f"❌ Error en la partida: {resultado.get('error', 'Error desconocido')}")
            return 1
        
        print("\\n✓ Evaluación completada")
        return 0
        
    except KeyboardInterrupt:
        print("\\n🛑 Partida cancelada por el usuario")
        return 0
    except Exception as e:
        print(f"\\n❌ Error durante la partida: {e}")
        logger.error("Error durante la evaluación: %s", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())