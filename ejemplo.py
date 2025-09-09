#!/usr/bin/env python3
"""
Ejemplo de uso básico del proyecto de comparación de agentes de ajedrez.

Este script demuestra cómo usar los diferentes componentes del proyecto
para entrenar agentes, evaluarlos y generar reportes.
"""

import os
import sys
import logging
from pathlib import Path

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ejemplo_entrenamiento_rapido():
    """
    Ejemplo de entrenamiento rápido con parámetros reducidos para pruebas.
    """
    print("=" * 50)
    print("EJEMPLO: ENTRENAMIENTO RÁPIDO")
    print("=" * 50)
    
    try:
        from entrenamiento.entrenar_dqn import main as entrenar_dqn
        from entrenamiento.entrenar_ppo import main as entrenar_ppo
        
        # Entrenar DQN con parámetros reducidos
        print("\n1. Entrenando agente DQN (versión rápida)...")
        sys.argv = ['entrenar_dqn.py', '--timesteps', '5000', '--save-freq', '1000']
        entrenar_dqn()
        
        # Entrenar PPO con parámetros reducidos
        print("\n2. Entrenando agente PPO (versión rápida)...")
        sys.argv = ['entrenar_ppo.py', '--timesteps', '5000', '--save-freq', '1000']
        entrenar_ppo()
        
        print("✓ Entrenamiento rápido completado")
        
    except Exception as e:
        print(f"✗ Error en entrenamiento: {e}")
        return False
    
    return True

def ejemplo_evaluacion_basica():
    """
    Ejemplo de evaluación básica entre agentes.
    """
    print("\n" + "=" * 50)
    print("EJEMPLO: EVALUACIÓN BÁSICA")
    print("=" * 50)
    
    try:
        from entrenamiento.evaluar_agentes import main as evaluar
        
        # Evaluación con pocas partidas para el ejemplo
        print("\nEvaluando agentes con 5 partidas por enfrentamiento...")
        sys.argv = ['evaluar_agentes.py', '--partidas', '5', '--timeout', '5']
        evaluar()
        
        print("✓ Evaluación básica completada")
        
    except Exception as e:
        print(f"✗ Error en evaluación: {e}")
        return False
    
    return True

def ejemplo_analisis_manual():
    """
    Ejemplo de análisis manual de resultados.
    """
    print("\n" + "=" * 50)
    print("EJEMPLO: ANÁLISIS MANUAL")
    print("=" * 50)
    
    try:
        from analisis.metricas_desempeno import calcular_metricas_desde_csv
        from analisis.visualizacion import crear_dashboard
        from utils import cargar_json
        
        # Buscar archivo de resultados más reciente
        resultados_dir = Path("resultados")
        archivos_csv = list(resultados_dir.glob("**/evaluacion_*.csv"))
        
        if not archivos_csv:
            print("No se encontraron archivos de evaluación. Ejecuta primero la evaluación.")
            return False
        
        archivo_mas_reciente = max(archivos_csv, key=os.path.getctime)
        print(f"Analizando: {archivo_mas_reciente}")
        
        # Calcular métricas
        metricas = calcular_metricas_desde_csv(str(archivo_mas_reciente))
        
        # Mostrar resumen
        print("\nRESUMEN DE RESULTADOS:")
        print("-" * 30)
        
        for agente, stats in metricas['por_agente'].items():
            print(f"{agente:12} | "
                  f"V:{stats['victorias']:2d} "
                  f"E:{stats['empates']:2d} "
                  f"D:{stats['derrotas']:2d} "
                  f"| Puntos: {stats['puntos']:.1f} "
                  f"| {stats['porcentaje']:.1f}%")
        
        # Crear visualizaciones
        print("\nGenerando gráficas...")
        crear_dashboard(metricas, "resultados/ejemplo_dashboard.png")
        
        print("✓ Análisis manual completado")
        print(f"Dashboard guardado en: resultados/ejemplo_dashboard.png")
        
    except Exception as e:
        print(f"✗ Error en análisis: {e}")
        return False
    
    return True

def ejemplo_partida_individual():
    """
    Ejemplo de una partida individual entre dos agentes.
    """
    print("\n" + "=" * 50)
    print("EJEMPLO: PARTIDA INDIVIDUAL")
    print("=" * 50)
    
    try:
        from entorno.entorno_ajedrez import EntornoAjedrez
        from agentes.agente_minimax import AgenteMinimax
        from agentes.agente_mcts import AgenteMCTS
        import time
        
        # Crear entorno y agentes
        entorno = EntornoAjedrez()
        agente_blanco = AgenteMinimax(profundidad=2, usar_alpha_beta=True)
        agente_negro = AgenteMCTS(num_simulaciones=100)
        
        print("Minimax (Blancas) vs MCTS (Negras)")
        print("Profundidad Minimax: 2, Simulaciones MCTS: 100")
        print("-" * 40)
        
        observacion = entorno.reset()
        movimiento_num = 1
        tiempo_inicio = time.time()
        
        while True:
            movimientos_validos = entorno.obtener_movimientos_validos()
            
            if not movimientos_validos or entorno.terminado:
                break
            
            # Seleccionar agente actual
            agente_actual = agente_blanco if entorno.turno_blancas else agente_negro
            nombre_agente = "Minimax" if entorno.turno_blancas else "MCTS"
            
            # Realizar movimiento
            inicio_mov = time.time()
            movimiento = agente_actual.seleccionar_movimiento(observacion, movimientos_validos)
            tiempo_mov = time.time() - inicio_mov
            
            # Aplicar movimiento
            observacion, recompensa, terminado, info = entorno.step(movimiento)
            
            # Mostrar progreso
            print(f"Mov {movimiento_num:2d}: {nombre_agente:7} | "
                  f"{movimiento} | {tiempo_mov:.2f}s")
            
            movimiento_num += 1
            
            # Límite de movimientos para el ejemplo
            if movimiento_num > 20:
                print("... (limitado a 20 movimientos para el ejemplo)")
                break
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Resultado
        print("-" * 40)
        if entorno.terminado:
            if info.get('ganador') == 'blancas':
                print("Resultado: Victoria de Minimax (Blancas)")
            elif info.get('ganador') == 'negras':
                print("Resultado: Victoria de MCTS (Negras)")
            else:
                print("Resultado: Empate")
        else:
            print("Partida interrumpida")
        
        print(f"Tiempo total: {tiempo_total:.2f}s")
        print(f"Movimientos jugados: {movimiento_num - 1}")
        
        print("✓ Partida individual completada")
        
    except Exception as e:
        print(f"✗ Error en partida individual: {e}")
        return False
    
    return True

def ejemplo_configuracion_personalizada():
    """
    Ejemplo de cómo personalizar la configuración.
    """
    print("\n" + "=" * 50)
    print("EJEMPLO: CONFIGURACIÓN PERSONALIZADA")
    print("=" * 50)
    
    try:
        from config import CONFIG_AGENTES, CONFIG_EVALUACION
        
        print("Configuración actual de agentes:")
        print("-" * 30)
        
        for agente, config in CONFIG_AGENTES.items():
            print(f"\n{agente.upper()}:")
            for param, valor in config.items():
                print(f"  {param}: {valor}")
        
        print(f"\nConfiguración de evaluación:")
        print("-" * 30)
        for param, valor in CONFIG_EVALUACION.items():
            print(f"  {param}: {valor}")
        
        print("\nPara personalizar:")
        print("1. Edita config_local.py")
        print("2. O modifica config.py directamente")
        print("3. O pasa parámetros por línea de comandos")
        
        print("✓ Ejemplo de configuración mostrado")
        
    except Exception as e:
        print(f"✗ Error en configuración: {e}")
        return False
    
    return True

def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("=" * 60)
    print("EJEMPLOS DE USO - CHESS-IA-2-AGENT-COMPARATION")
    print("=" * 60)
    
    ejemplos = [
        ("Configuración personalizada", ejemplo_configuracion_personalizada),
        ("Partida individual", ejemplo_partida_individual),
        ("Entrenamiento rápido", ejemplo_entrenamiento_rapido),
        ("Evaluación básica", ejemplo_evaluacion_basica),
        ("Análisis manual", ejemplo_analisis_manual),
    ]
    
    print("Ejemplos disponibles:")
    for i, (nombre, _) in enumerate(ejemplos, 1):
        print(f"{i}. {nombre}")
    
    print("\nEjecutando todos los ejemplos...")
    print("(Presiona Ctrl+C para interrumpir)")
    
    try:
        for nombre, funcion in ejemplos:
            print(f"\n>>> Ejecutando: {nombre}")
            if not funcion():
                print(f">>> ✗ Falló: {nombre}")
                break
            print(f">>> ✓ Completado: {nombre}")
            
            # Pausa entre ejemplos
            print("    (Pausa de 2 segundos...)")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n>>> Ejemplos interrumpidos por el usuario")
    
    except Exception as e:
        print(f"\n\n>>> Error general: {e}")
    
    finally:
        print("\n" + "=" * 60)
        print("EJEMPLOS FINALIZADOS")
        print("=" * 60)
        print("\nPara uso normal del proyecto:")
        print("  python principal.py --help")
        print("\nPara entrenamiento completo:")
        print("  python principal.py --entrenar dqn --entrenar ppo")
        print("  python principal.py --evaluar --reporte")

if __name__ == "__main__":
    import time
    main()
