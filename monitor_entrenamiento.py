"""
Script para monitorear el progreso del entrenamiento de AlphaZero.

Muestra información sobre:
- Modelos guardados
- Logs recientes
- Uso de recursos
"""

import os
import time
from datetime import datetime
import glob

def formato_tamano(bytes):
    """Convierte bytes a formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def mostrar_modelos():
    """Muestra los modelos guardados."""
    print("\n" + "=" * 70)
    print("MODELOS GUARDADOS")
    print("=" * 70)
    
    modelos_dir = "modelos"
    if not os.path.exists(modelos_dir):
        print("⚠ Directorio 'modelos/' no existe aún")
        return
    
    archivos = glob.glob(os.path.join(modelos_dir, "*.pth"))
    
    if not archivos:
        print("⚠ No hay modelos guardados aún")
        print("  (Los modelos se guardan cada 10 iteraciones)")
        return
    
    archivos.sort(key=os.path.getmtime)
    
    print(f"\nTotal de modelos: {len(archivos)}")
    print("\nÚltimos modelos guardados:")
    print("-" * 70)
    
    for archivo in archivos[-5:]:  # Mostrar últimos 5
        nombre = os.path.basename(archivo)
        tamano = os.path.getsize(archivo)
        modificado = os.path.getmtime(archivo)
        fecha = datetime.fromtimestamp(modificado).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"📦 {nombre}")
        print(f"   Tamaño: {formato_tamano(tamano)}")
        print(f"   Fecha: {fecha}")
        print()

def estimar_progreso():
    """Estima el progreso basándose en los modelos guardados."""
    print("\n" + "=" * 70)
    print("ESTIMACIÓN DE PROGRESO")
    print("=" * 70)
    
    modelos = glob.glob("modelos/alphazero_iter_*.pth")
    
    if not modelos:
        print("\n⏳ Entrenamiento en progreso...")
        print("   Esperando primer checkpoint (iteración 10)")
        return
    
    # Extraer números de iteración
    iteraciones = []
    for modelo in modelos:
        nombre = os.path.basename(modelo)
        try:
            iter_num = int(nombre.split('_')[2].split('.')[0])
            iteraciones.append(iter_num)
        except:
            pass
    
    if iteraciones:
        max_iter = max(iteraciones)
        print(f"\n✓ Última iteración guardada: {max_iter}")
        
        # Estimar progreso para diferentes configuraciones comunes
        configs = [
            ("Rápido (10 iter)", 10),
            ("Corto (50 iter)", 50),
            ("Normal (100 iter)", 100),
            ("Largo (200 iter)", 200)
        ]
        
        print("\nProgreso estimado:")
        for nombre, total in configs:
            porcentaje = (max_iter / total) * 100
            barra_len = 30
            barra = "█" * int(barra_len * porcentaje / 100)
            espacios = " " * (barra_len - len(barra))
            print(f"  {nombre:20s} [{barra}{espacios}] {porcentaje:.1f}%")

def verificar_proceso():
    """Verifica si hay un proceso de entrenamiento activo."""
    print("\n" + "=" * 70)
    print("INFORMACIÓN DEL SISTEMA")
    print("=" * 70)
    
    try:
        import psutil
        
        # Buscar procesos de Python ejecutando entrenamiento
        procesos_python = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'alphazero' in cmdline.lower() or 'entrenar' in cmdline.lower():
                        procesos_python.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline[:80] + '...' if len(cmdline) > 80 else cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if procesos_python:
            print("\n✓ Procesos de entrenamiento detectados:")
            for proc in procesos_python:
                print(f"  PID {proc['pid']}: {proc['cmdline']}")
        else:
            print("\n⚠ No se detectaron procesos de entrenamiento activos")
        
        # Información del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memoria = psutil.virtual_memory()
        
        print(f"\nUso de recursos:")
        print(f"  CPU: {cpu_percent}%")
        print(f"  RAM: {memoria.percent}% ({formato_tamano(memoria.used)}/{formato_tamano(memoria.total)})")
        
    except ImportError:
        print("\n⚠ psutil no instalado - no se puede mostrar info de procesos")
        print("  Instalar con: pip install psutil")

def mostrar_consejos():
    """Muestra consejos útiles."""
    print("\n" + "=" * 70)
    print("CONSEJOS")
    print("=" * 70)
    print("""
📊 MONITOREO:
   - Ejecuta este script periódicamente para ver el progreso
   - Los modelos se guardan cada 10 iteraciones por defecto

⚡ SI EL ENTRENAMIENTO ESTÁ LENTO:
   - Reduce num_simulaciones (ej: 400 en vez de 800)
   - Reduce num_episodios (ej: 50 en vez de 100)
   - Usa el script: python entrenar_rapido_alphazero.py

🛑 DETENER ENTRENAMIENTO:
   - Presiona Ctrl+C en la terminal del entrenamiento
   - El último modelo guardado estará disponible

🎮 PROBAR EL MODELO:
   - python ver_alphazero_jugar.py
   - Usará el modelo más reciente automáticamente

📈 CONTINUAR ENTRENAMIENTO:
   - Puedes entrenar más iteraciones después
   - El modelo se actualizará progresivamente
""")

def main():
    """Función principal."""
    print("=" * 70)
    print("  MONITOR DE ENTRENAMIENTO - ALPHAZERO")
    print("=" * 70)
    print(f"\nFecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mostrar_modelos()
    estimar_progreso()
    verificar_proceso()
    mostrar_consejos()
    
    print("\n" + "=" * 70)
    print("Ejecuta este script nuevamente para actualizar la información")
    print("=" * 70)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
