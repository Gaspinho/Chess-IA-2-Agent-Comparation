#!/usr/bin/env python3
"""
Script de instalación y configuración inicial para el proyecto de comparación de agentes de ajedrez.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def ejecutar_comando(comando, descripcion=""):
    """
    Ejecuta un comando y maneja errores.
    
    Args:
        comando: Lista con el comando a ejecutar
        descripcion: Descripción del comando para logging
    """
    try:
        print(f"Ejecutando: {descripcion or ' '.join(comando)}")
        resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
        print(f"✓ Completado: {descripcion}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error en {descripcion}: {e}")
        print(f"Salida del error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Comando no encontrado: {comando[0]}")
        return False

def verificar_python():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Se requiere Python 3.8 o superior")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def crear_directorios():
    """Crea los directorios necesarios del proyecto."""
    directorios = [
        'modelos',
        'resultados',
        'logs',
        'resultados/graficas',
        'resultados/reportes',
        'resultados/partidas'
    ]
    
    for directorio in directorios:
        Path(directorio).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directorio creado: {directorio}")

def instalar_dependencias():
    """Instala las dependencias del proyecto."""
    print("Instalando dependencias...")
    
    # Actualizar pip
    if not ejecutar_comando([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                           "Actualizando pip"):
        return False
    
    # Instalar dependencias desde requirements.txt
    if os.path.exists("requerimientos.txt"):
        if not ejecutar_comando([sys.executable, "-m", "pip", "install", "-r", "requerimientos.txt"],
                               "Instalando dependencias desde requerimientos.txt"):
            return False
    else:
        # Instalar dependencias manualmente
        dependencias = [
            "pettingzoo[classic]",
            "stable-baselines3[extra]",
            "torch",
            "gymnasium",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "tqdm",
            "chess",
            "tensorboard"
        ]
        
        for dep in dependencias:
            if not ejecutar_comando([sys.executable, "-m", "pip", "install", dep],
                                   f"Instalando {dep}"):
                print(f"Advertencia: No se pudo instalar {dep}")
    
    return True

def verificar_instalacion():
    """Verifica que todas las dependencias estén instaladas correctamente."""
    print("Verificando instalación...")
    
    modulos_requeridos = [
        'pettingzoo',
        'stable_baselines3',
        'torch',
        'gymnasium',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'chess'
    ]
    
    modulos_faltantes = []
    
    for modulo in modulos_requeridos:
        try:
            __import__(modulo)
            print(f"✓ {modulo}")
        except ImportError:
            print(f"✗ {modulo} - NO INSTALADO")
            modulos_faltantes.append(modulo)
    
    if modulos_faltantes:
        print(f"\nMódulos faltantes: {', '.join(modulos_faltantes)}")
        return False
    
    print("✓ Todas las dependencias están instaladas")
    return True

def crear_archivo_config():
    """Crea un archivo de configuración local si no existe."""
    config_local = "config_local.py"
    
    if not os.path.exists(config_local):
        contenido = '''# Configuración local - personaliza estos valores según tus necesidades

from config import *

# Sobrescribir configuraciones específicas aquí
# Ejemplo:
# CONFIG_AGENTES['dqn']['total_timesteps'] = 50000  # Reducir para pruebas rápidas
# CONFIG_EVALUACION['num_partidas'] = 10  # Menos partidas para pruebas

# Configuración de hardware
USE_GPU = True  # Cambiar a False si no tienes GPU compatible

# Rutas personalizadas (opcional)
# RUTA_MODELOS_CUSTOM = "/ruta/personalizada/modelos"
# RUTA_RESULTADOS_CUSTOM = "/ruta/personalizada/resultados"

print("Configuración local cargada")
'''
        
        with open(config_local, 'w', encoding='utf-8') as f:
            f.write(contenido)
        
        print(f"✓ Archivo de configuración local creado: {config_local}")
        print("  Puedes editarlo para personalizar la configuración")

def ejecutar_prueba_rapida():
    """Ejecuta una prueba rápida del sistema."""
    print("Ejecutando prueba rápida del sistema...")
    
    try:
        # Importar módulos principales
        from entorno.entorno_ajedrez import EntornoAjedrez
        from agentes.agente_minimax import AgenteMinimax
        
        # Crear entorno y agente
        entorno = EntornoAjedrez()
        agente = AgenteMinimax(profundidad=1)
        
        # Ejecutar algunos movimientos
        observacion = entorno.reset()
        for _ in range(3):
            movimientos_validos = entorno.obtener_movimientos_validos()
            if not movimientos_validos:
                break
            
            movimiento = agente.seleccionar_movimiento(observacion, movimientos_validos)
            observacion, recompensa, terminado, info = entorno.step(movimiento)
            
            if terminado:
                break
        
        print("✓ Prueba rápida completada exitosamente")
        return True
        
    except Exception as e:
        print(f"✗ Error en la prueba rápida: {e}")
        return False

def main():
    """Función principal de instalación."""
    print("=" * 60)
    print("INSTALACIÓN DEL PROYECTO CHESS-IA-2-AGENT-COMPARATION")
    print("=" * 60)
    
    # Verificar Python
    if not verificar_python():
        return False
    
    # Crear directorios
    print("\n1. Creando estructura de directorios...")
    crear_directorios()
    
    # Instalar dependencias
    print("\n2. Instalando dependencias...")
    if not instalar_dependencias():
        print("✗ Error al instalar dependencias")
        return False
    
    # Verificar instalación
    print("\n3. Verificando instalación...")
    if not verificar_instalacion():
        print("✗ Verificación fallida")
        return False
    
    # Crear configuración local
    print("\n4. Creando configuración local...")
    crear_archivo_config()
    
    # Prueba rápida
    print("\n5. Ejecutando prueba rápida...")
    if not ejecutar_prueba_rapida():
        print("✗ Prueba rápida fallida")
        return False
    
    print("\n" + "=" * 60)
    print("✓ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("\nPróximos pasos:")
    print("1. Revisar y ajustar config_local.py si es necesario")
    print("2. Ejecutar: python principal.py --entrenar dqn")
    print("3. Ejecutar: python principal.py --entrenar ppo")
    print("4. Ejecutar: python principal.py --evaluar")
    print("5. Ejecutar: python principal.py --reporte")
    print("\nPara ayuda completa: python principal.py --help")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        print("\n✗ La instalación falló. Revisa los errores anteriores.")
        sys.exit(1)
    else:
        print("\n✓ Todo listo para comenzar!")
        sys.exit(0)
