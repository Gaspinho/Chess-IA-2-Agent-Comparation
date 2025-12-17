"""
Script para ejecutar partidas entre AlphaZero y PPO.

Este script permite ejecutar múltiples partidas entre los agentes
AlphaZero y PPO, alternando los colores y guardando estadísticas.

Uso:
  python ejecutar_alphazero_vs_ppo.py
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chess
import numpy as np
from agentes.agente_alphazero import AgenteAlphaZero
from agentes.agente_ppo import AgentePPO, ChessGymWrapper
from entorno.entorno_ajedrez import EntornoAjedrez

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluadorAlphaZeroPPO:
    """
    Clase para evaluar partidas entre AlphaZero y PPO.
    """
    
    def __init__(self, 
                 timeout_jugada: float = 30.0,
                 max_jugadas: int = 200):
        """
        Inicializa el evaluador.
        
        Args:
            timeout_jugada: Tiempo máximo por jugada en segundos
            max_jugadas: Número máximo de jugadas antes de declarar empate
        """
        self.timeout_jugada = timeout_jugada
        self.max_jugadas = max_jugadas
        
    def jugar_partida(self, 
                     agente_blancas: Any,
                     agente_negras: Any,
                     nombre_blancas: str,
                     nombre_negras: str,
                     mostrar_progreso: bool = True) -> Dict[str, Any]:
        """
        Ejecuta una partida entre dos agentes.
        
        Args:
            agente_blancas: Agente que juega con blancas
            agente_negras: Agente que juega con negras
            nombre_blancas: Nombre del agente blancas
            nombre_negras: Nombre del agente negras
            mostrar_progreso: Si se debe mostrar el progreso de la partida
            
        Returns:
            Diccionario con resultados de la partida
        """
        inicio_partida = time.time()
        
        # Crear entorno
        entorno = EntornoAjedrez()
        observacion = entorno.reiniciar()
        
        jugadas = []
        tiempo_blancas = 0.0
        tiempo_negras = 0.0
        terminado = False
        num_jugada = 0
        
        if mostrar_progreso:
            print(f"  Iniciando partida: {nombre_blancas} (Blancas) vs {nombre_negras} (Negras)")
        
        while not terminado and num_jugada < self.max_jugadas:
            num_jugada += 1
            
            # Determinar agente actual
            es_turno_blancas = entorno.tablero.turn == chess.WHITE
            agente_actual = agente_blancas if es_turno_blancas else agente_negras
            nombre_actual = nombre_blancas if es_turno_blancas else nombre_negras
            
            # Obtener acciones legales
            acciones_legales = entorno.obtener_acciones_legales()
            
            if len(acciones_legales) == 0:
                terminado = True
                break
            
            # Seleccionar acción
            try:
                inicio_jugada = time.time()
                # Crear info con tablero y estado del juego
                info = {
                    'tablero': entorno.obtener_tablero(),
                    'turno': entorno.tablero.turn,
                    'movimientos_legales': entorno.obtener_movimientos_legales()
                }
                accion = agente_actual.seleccionar_accion(observacion, acciones_legales, info)
                tiempo_jugada = time.time() - inicio_jugada
                
                if es_turno_blancas:
                    tiempo_blancas += tiempo_jugada
                else:
                    tiempo_negras += tiempo_jugada
                
                # Verificar timeout
                if tiempo_jugada > self.timeout_jugada:
                    logger.warning(f"Timeout en jugada {num_jugada} para {nombre_actual}")
                
                # Ejecutar acción
                observacion, recompensa, terminado, info = entorno.paso(accion)
                
                # Guardar jugada
                if info.get('movimiento_realizado'):
                    jugadas.append(info['movimiento_realizado'])
                    
                    if mostrar_progreso and num_jugada % 10 == 0:
                        print(f"    Jugada {num_jugada}: {info['movimiento_realizado']} ({nombre_actual}, {tiempo_jugada:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error en jugada {num_jugada}: {str(e)}")
                terminado = True
                break
        
        tiempo_total = time.time() - inicio_partida
        
        # Determinar resultado
        resultado_tablero = entorno.tablero.result()
        tipo_fin = self._determinar_tipo_fin(entorno.tablero)
        
        if resultado_tablero == "1-0":
            ganador = nombre_blancas
        elif resultado_tablero == "0-1":
            ganador = nombre_negras
        else:
            ganador = "Empate"
        
        resultado = {
            'exitoso': True,
            'ganador': ganador,
            'resultado_tablero': resultado_tablero,
            'tipo_fin': tipo_fin,
            'num_jugadas': num_jugada,
            'tiempo_total': tiempo_total,
            'tiempo_blancas': tiempo_blancas,
            'tiempo_negras': tiempo_negras,
            'jugadas': [str(m) for m in jugadas],
            'pgn': self._generar_pgn(nombre_blancas, nombre_negras, jugadas, resultado_tablero)
        }
        
        if mostrar_progreso:
            print(f"  ✓ Partida completada - Ganador: {ganador} ({num_jugada} jugadas, {tiempo_total:.1f}s)")
        
        return resultado
    
    def _determinar_tipo_fin(self, tablero: chess.Board) -> str:
        """Determina el tipo de finalización de la partida."""
        if tablero.is_checkmate():
            return "jaque_mate"
        elif tablero.is_stalemate():
            return "tablas_ahogado"
        elif tablero.is_insufficient_material():
            return "tablas_material_insuficiente"
        elif tablero.can_claim_fifty_moves():
            return "tablas_50_movimientos"
        elif tablero.can_claim_threefold_repetition():
            return "tablas_repeticion"
        else:
            return "otro"
    
    def _generar_pgn(self, blancas: str, negras: str, jugadas: List, resultado: str) -> str:
        """Genera el PGN de la partida."""
        pgn_lines = [
            f'[Event "AlphaZero vs PPO"]',
            f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]',
            f'[White "{blancas}"]',
            f'[Black "{negras}"]',
            f'[Result "{resultado}"]',
            ''
        ]
        
        # Agregar jugadas
        jugadas_str = []
        for i, jugada in enumerate(jugadas):
            if i % 2 == 0:
                jugadas_str.append(f"{i//2 + 1}. {jugada}")
            else:
                jugadas_str[-1] += f" {jugada}"
        
        pgn_lines.append(" ".join(jugadas_str))
        pgn_lines.append(resultado)
        
        return "\n".join(pgn_lines)


def ejecutar_torneo(num_partidas: int = 10,
                   num_simulaciones_alphazero: int = 400,
                   modelo_alphazero_path: str = None,
                   modelo_ppo_path: str = None):
    """
    Ejecuta un torneo entre AlphaZero y PPO.
    
    Args:
        num_partidas: Número total de partidas a jugar
        num_simulaciones_alphazero: Simulaciones MCTS para AlphaZero
        modelo_alphazero_path: Ruta al modelo pre-entrenado de AlphaZero
        modelo_ppo_path: Ruta al modelo pre-entrenado de PPO
    """
    print("=" * 70)
    print("  TORNEO: AlphaZero vs PPO")
    print("=" * 70)
    print()
    
    # Crear directorio de resultados si no existe
    dir_resultados = os.path.join(os.path.dirname(__file__), 'resultados')
    os.makedirs(dir_resultados, exist_ok=True)
    
    # Inicializar agentes
    print("Inicializando agentes...")
    print(f"  - AlphaZero: {num_simulaciones_alphazero} simulaciones MCTS")
    
    try:
        agente_alphazero = AgenteAlphaZero(
            num_simulaciones=num_simulaciones_alphazero,
            temperatura=0.1,  # Baja temperatura para juego más determinista
            modelo_path=modelo_alphazero_path
        )
        print("  ✓ AlphaZero inicializado")
    except Exception as e:
        print(f"  ✗ Error inicializando AlphaZero: {e}")
        return
    
    try:
        # Crear entorno temporal para PPO con wrapper Gymnasium
        env_temp = EntornoAjedrez()
        env_gym = ChessGymWrapper(env_temp)
        agente_ppo = AgentePPO()
        
        # Cargar modelo PPO si existe
        if modelo_ppo_path and os.path.exists(modelo_ppo_path):
            agente_ppo.cargar(modelo_ppo_path)
            print(f"  ✓ PPO inicializado (modelo cargado: {modelo_ppo_path})")
        else:
            # Crear modelo con configuración por defecto
            agente_ppo.crear_modelo(env_gym)
            print("  ✓ PPO inicializado (sin modelo pre-entrenado)")
    except Exception as e:
        print(f"  ✗ Error inicializando PPO: {e}")
        logger.error("Error detallado PPO: %s", str(e), exc_info=True)
        return
    
    print()
    
    # Crear evaluador
    evaluador = EvaluadorAlphaZeroPPO(
        timeout_jugada=60.0,
        max_jugadas=150
    )
    
    # Estadísticas globales
    estadisticas = {
        'victorias_alphazero': 0,
        'victorias_ppo': 0,
        'empates': 0,
        'partidas': [],
        'tiempo_total': 0.0,
        'configuracion': {
            'num_partidas': num_partidas,
            'num_simulaciones_alphazero': num_simulaciones_alphazero,
            'modelo_alphazero': modelo_alphazero_path or "No cargado",
            'modelo_ppo': modelo_ppo_path or "No cargado"
        }
    }
    
    inicio_torneo = time.time()
    
    # Ejecutar partidas alternando colores
    for i in range(num_partidas):
        print(f"\n{'='*70}")
        print(f"  PARTIDA {i+1}/{num_partidas}")
        print(f"{'='*70}")
        
        # Alternar colores
        if i % 2 == 0:
            # AlphaZero juega con blancas
            resultado = evaluador.jugar_partida(
                agente_blancas=agente_alphazero,
                agente_negras=agente_ppo,
                nombre_blancas="AlphaZero",
                nombre_negras="PPO",
                mostrar_progreso=True
            )
        else:
            # PPO juega con blancas
            resultado = evaluador.jugar_partida(
                agente_blancas=agente_ppo,
                agente_negras=agente_alphazero,
                nombre_blancas="PPO",
                nombre_negras="AlphaZero",
                mostrar_progreso=True
            )
        
        # Actualizar estadísticas
        estadisticas['partidas'].append(resultado)
        
        if resultado['ganador'] == 'AlphaZero':
            estadisticas['victorias_alphazero'] += 1
        elif resultado['ganador'] == 'PPO':
            estadisticas['victorias_ppo'] += 1
        else:
            estadisticas['empates'] += 1
        
        # Mostrar estadísticas parciales
        print(f"\n  Estadísticas parciales:")
        print(f"    AlphaZero: {estadisticas['victorias_alphazero']} victorias")
        print(f"    PPO: {estadisticas['victorias_ppo']} victorias")
        print(f"    Empates: {estadisticas['empates']}")
    
    tiempo_total_torneo = time.time() - inicio_torneo
    estadisticas['tiempo_total'] = tiempo_total_torneo
    
    # Mostrar resultados finales
    print("\n" + "=" * 70)
    print("  RESULTADOS FINALES DEL TORNEO")
    print("=" * 70)
    print(f"\n  Total de partidas: {num_partidas}")
    print(f"  AlphaZero: {estadisticas['victorias_alphazero']} victorias ({estadisticas['victorias_alphazero']/num_partidas*100:.1f}%)")
    print(f"  PPO: {estadisticas['victorias_ppo']} victorias ({estadisticas['victorias_ppo']/num_partidas*100:.1f}%)")
    print(f"  Empates: {estadisticas['empates']} ({estadisticas['empates']/num_partidas*100:.1f}%)")
    print(f"\n  Tiempo total: {tiempo_total_torneo/60:.1f} minutos")
    print(f"  Tiempo promedio por partida: {tiempo_total_torneo/num_partidas:.1f} segundos")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_resultados = os.path.join(dir_resultados, f'alphazero_vs_ppo_{timestamp}.json')
    
    # Preparar datos para guardar (sin objetos complejos)
    datos_guardar = {
        'configuracion': estadisticas['configuracion'],
        'resumen': {
            'total_partidas': num_partidas,
            'victorias_alphazero': estadisticas['victorias_alphazero'],
            'victorias_ppo': estadisticas['victorias_ppo'],
            'empates': estadisticas['empates'],
            'tiempo_total': tiempo_total_torneo
        },
        'partidas': [
            {
                'numero': idx + 1,
                'ganador': p['ganador'],
                'resultado_tablero': p['resultado_tablero'],
                'tipo_fin': p['tipo_fin'],
                'num_jugadas': p['num_jugadas'],
                'tiempo_total': p['tiempo_total'],
                'tiempo_blancas': p['tiempo_blancas'],
                'tiempo_negras': p['tiempo_negras']
            }
            for idx, p in enumerate(estadisticas['partidas'])
        ]
    }
    
    with open(archivo_resultados, 'w', encoding='utf-8') as f:
        json.dump(datos_guardar, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Resultados guardados en: {archivo_resultados}")
    
    # Guardar PGN de las partidas
    archivo_pgn = os.path.join(dir_resultados, f'alphazero_vs_ppo_{timestamp}.pgn')
    with open(archivo_pgn, 'w', encoding='utf-8') as f:
        for idx, partida in enumerate(estadisticas['partidas']):
            f.write(f"\n; Partida {idx + 1}\n")
            f.write(partida['pgn'])
            f.write("\n\n")
    
    print(f"  ✓ PGN guardado en: {archivo_pgn}")
    print()


def main():
    """Función principal."""
    print("\n" + "=" * 70)
    print("  CONFIGURACIÓN DEL TORNEO AlphaZero vs PPO")
    print("=" * 70)
    print()
    
    # Verificar dependencias
    try:
        import chess
        import torch
        import numpy as np
        print("✓ Dependencias verificadas")
    except ImportError as e:
        print(f"✗ Error: Dependencia faltante - {e}")
        return 1
    
    print()
    print("Configuraciones sugeridas:")
    print("1. Rápido: 5 partidas, 200 simulaciones (~5-10 minutos)")
    print("2. Normal: 10 partidas, 400 simulaciones (~15-30 minutos)")
    print("3. Completo: 20 partidas, 800 simulaciones (~1-2 horas)")
    print("4. Personalizado")
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
    if opcion_num == 1:
        num_partidas = 5
        num_simulaciones = 200
    elif opcion_num == 2:
        num_partidas = 10
        num_simulaciones = 400
    elif opcion_num == 3:
        num_partidas = 20
        num_simulaciones = 800
    else:
        # Personalizado
        while True:
            try:
                num_partidas = int(input("Número de partidas [10]: ").strip() or "10")
                if num_partidas > 0:
                    break
                print("Debe ser mayor a 0")
            except ValueError:
                print("Por favor, ingresa un número válido")
        
        while True:
            try:
                num_simulaciones = int(input("Simulaciones AlphaZero [400]: ").strip() or "400")
                if num_simulaciones > 0:
                    break
                print("Debe ser mayor a 0")
            except ValueError:
                print("Por favor, ingresa un número válido")
    
    print()
    print(f"Configuración seleccionada:")
    print(f"  - Número de partidas: {num_partidas}")
    print(f"  - Simulaciones AlphaZero: {num_simulaciones}")
    print()
    
    # Preguntar por modelos pre-entrenados
    print("¿Deseas cargar modelos pre-entrenados? (opcional)")
    
    modelo_alphazero_path = None
    usar_modelo_az = input("Cargar modelo AlphaZero? (s/N): ").strip().lower()
    if usar_modelo_az == 's':
        path = input("Ruta al modelo AlphaZero (.pth): ").strip()
        if path and os.path.exists(path):
            modelo_alphazero_path = path
            print(f"  ✓ Modelo AlphaZero: {path}")
        else:
            print("  ⚠ Archivo no encontrado, se usará modelo sin entrenar")
    
    modelo_ppo_path = None
    usar_modelo_ppo = input("Cargar modelo PPO? (s/N): ").strip().lower()
    if usar_modelo_ppo == 's':
        path = input("Ruta al modelo PPO (.zip): ").strip()
        if path and os.path.exists(path):
            modelo_ppo_path = path
            print(f"  ✓ Modelo PPO: {path}")
        else:
            print("  ⚠ Archivo no encontrado, se usará modelo sin entrenar")
    
    print()
    input("Presiona ENTER para comenzar el torneo...")
    print()
    
    # Ejecutar torneo
    try:
        ejecutar_torneo(
            num_partidas=num_partidas,
            num_simulaciones_alphazero=num_simulaciones,
            modelo_alphazero_path=modelo_alphazero_path,
            modelo_ppo_path=modelo_ppo_path
        )
        
        print("\n✓ Torneo completado exitosamente")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Torneo cancelado por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error durante el torneo: {e}")
        logger.error("Error durante el torneo: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
