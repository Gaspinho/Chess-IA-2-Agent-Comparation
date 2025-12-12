"""
Script de entrenamiento RÁPIDO para AlphaZero.

Este script usa configuraciones reducidas para ver progreso rápidamente
y verificar que el entrenamiento funciona correctamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from entrenamiento.entrenar_alphazero import entrenar_agente_alphazero
import logging

# Configurar logging en modo verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("  ENTRENAMIENTO RÁPIDO DE ALPHAZERO (MODO DEBUG)")
print("=" * 70)
print()
print("Esta configuración usa parámetros reducidos para:")
print("  - Ver progreso inmediato")
print("  - Verificar que el sistema funciona")
print("  - Entrenar un modelo básico rápidamente")
print()
print("Configuración:")
print("  - 5 iteraciones")
print("  - 5 episodios por iteración")
print("  - 100 simulaciones MCTS por jugada")
print("  - Batch size: 16")
print()
print("Tiempo estimado: 10-20 minutos en CPU")
print("=" * 70)
print()

input("Presiona ENTER para comenzar o Ctrl+C para cancelar...")
print()

# Configuración ULTRA-RÁPIDA para debug
config_rapida = {
    'learning_rate': 0.001,
    'num_simulaciones': 100,        # Reducido de 800
    'batch_size': 16,                # Reducido de 64
    'num_iteraciones': 5,            # Reducido de 100
    'num_episodios': 5,              # Reducido de 100
    'temperatura': 1.0,
    'c_puct': 1.5,
    'frecuencia_guardado': 2,        # Guardar cada 2 iteraciones
    'ruta_guardado': 'modelos/'
}

print("Iniciando entrenamiento...")
print()

resultado = entrenar_agente_alphazero(config_rapida)

print()
print("=" * 70)
print("RESULTADO DEL ENTRENAMIENTO")
print("=" * 70)

if resultado.get('entrenamiento_exitoso', False):
    print("✓ Entrenamiento completado exitosamente!")
    print()
    print(f"Iteraciones completadas: {resultado.get('iteraciones_completadas', 0)}")
    print(f"Tiempo total: {resultado.get('tiempo_total', 0):.2f} segundos")
    print(f"Pérdida final: {resultado.get('perdida_final', 0):.4f}")
    print(f"Modelo guardado en: {resultado.get('ruta_modelo', 'N/A')}")
    print()
    print("Próximos pasos:")
    print("  1. Ver el modelo jugar: python ver_alphazero_jugar.py")
    print("  2. Entrenar más: python principal.py --entrenar alphazero --num-iteraciones 50")
else:
    print("✗ Error en el entrenamiento:")
    print(f"  {resultado.get('error', 'Error desconocido')}")

print("=" * 70)
