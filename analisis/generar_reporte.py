"""
Generador de reportes automáticos en Markdown.
"""

import pandas as pd
import os
from typing import Dict, Any
from datetime import datetime
from .metricas_desempeno import CalculadorMetricas, generar_tabla_comparativa
from .visualizacion import GeneradorVisualizaciones


class GeneradorReporte:
    """Genera reportes automáticos en formato Markdown."""
    
    def generar_reporte_completo(self, archivo_datos: str, archivo_salida: str) -> bool:
        """
        Genera un reporte completo de análisis.
        
        Args:
            archivo_datos: Ruta del archivo CSV con resultados
            archivo_salida: Ruta del archivo Markdown de salida
            
        Returns:
            True si se generó exitosamente
        """
        try:
            # Cargar datos
            datos = pd.read_csv(archivo_datos)
            
            # Calcular métricas
            calculador = CalculadorMetricas(datos)
            metricas_generales = calculador.calcular_metricas_generales()
            metricas_agentes = calculador.calcular_metricas_por_agente()
            tabla_comparativa = generar_tabla_comparativa(metricas_agentes)
            
            # Generar visualizaciones
            visualizador = GeneradorVisualizaciones(datos)
            visualizador.generar_todas_visualizaciones("resultados/")
            
            # Generar reporte
            contenido = self._generar_contenido_markdown(
                metricas_generales, tabla_comparativa, metricas_agentes
            )
            
            # Guardar archivo
            os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                f.write(contenido)
            
            return True
            
        except Exception as e:
            print(f"Error al generar reporte: {e}")
            return False
    
    def _generar_contenido_markdown(self, metricas_generales: Dict[str, Any], 
                                   tabla_comparativa: pd.DataFrame,
                                   metricas_agentes: Dict[str, Dict[str, float]]) -> str:
        """Genera el contenido del reporte en Markdown."""
        
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        contenido = f"""# Reporte de Análisis - Agentes de Ajedrez

**Fecha de generación:** {fecha}

## 📊 Resumen Ejecutivo

Este reporte presenta los resultados de la evaluación comparativa entre cuatro agentes de ajedrez diferentes:
- **Minimax**: Algoritmo de búsqueda con poda alfa-beta
- **MCTS**: Monte Carlo Tree Search
- **DQN**: Deep Q-Network (Aprendizaje por Refuerzo)
- **PPO**: Proximal Policy Optimization (Aprendizaje por Refuerzo)

## 📈 Métricas Generales

- **Total de partidas jugadas:** {metricas_generales['total_partidas']:,}
- **Agentes evaluados:** {metricas_generales['agentes_evaluados']}
- **Tiempo promedio por partida:** {metricas_generales['tiempo_promedio_partida']:.2f} segundos
- **Jugadas promedio por partida:** {metricas_generales['jugadas_promedio']:.1f}
- **Tasa de completación:** {metricas_generales['tasa_completacion']:.1%}

## 🏆 Tabla Comparativa de Rendimiento

{tabla_comparativa.to_markdown(index=False)}

## 📊 Análisis Detallado por Agente

"""
        
        # Análisis por agente
        agentes_ordenados = sorted(metricas_agentes.keys(), 
                                 key=lambda x: metricas_agentes[x]['tasa_victoria'], 
                                 reverse=True)
        
        for i, agente in enumerate(agentes_ordenados, 1):
            stats = metricas_agentes[agente]
            contenido += f"""### {i}. {agente}

- **Tasa de victoria:** {stats['tasa_victoria']:.1%}
- **Partidas jugadas:** {stats['total_partidas']}
- **Victorias:** {stats['victorias']}
- **Empates:** {stats['empates']}
- **Derrotas:** {stats['derrotas']}
- **Tiempo promedio por jugada:** {stats['tiempo_promedio_jugada']:.3f} segundos
- **Recompensa promedio:** {stats['recompensa_promedio']:.3f}

"""
        
        contenido += """## 📊 Visualizaciones

### Gráfico de Victorias por Agente
![Victorias por Agente](../resultados/victorias_por_agente.png)

### Matriz de Enfrentamientos
![Matriz de Enfrentamientos](../resultados/matriz_enfrentamientos.png)

### Tiempo Promedio por Jugada
![Tiempo por Jugada](../resultados/tiempo_por_jugada.png)

## 🎯 Conclusiones y Recomendaciones

### Principales Hallazgos:
"""
        
        # Generar conclusiones automáticas
        mejor_agente = agentes_ordenados[0]
        peor_agente = agentes_ordenados[-1]
        
        contenido += f"""
1. **{mejor_agente}** demostró ser el agente más efectivo con una tasa de victoria del {metricas_agentes[mejor_agente]['tasa_victoria']:.1%}.

2. **{peor_agente}** obtuvo el menor rendimiento con una tasa de victoria del {metricas_agentes[peor_agente]['tasa_victoria']:.1%}.

3. El tiempo promedio por jugada varió significativamente entre agentes, reflejando diferentes complejidades algorítmicas.

### Recomendaciones:

- **Para aplicaciones en tiempo real:** Considerar el balance entre precisión y velocidad de respuesta.
- **Para competiciones:** Utilizar el agente con mayor tasa de victoria ({mejor_agente}).
- **Para entrenamiento:** Analizar las estrategias del mejor agente para mejorar otros algoritmos.

## 🔧 Metodología

- **Enfrentamientos:** Cada par de agentes jugó múltiples partidas alternando colores.
- **Métricas:** Se midieron victorias, empates, tiempo por jugada y recompensas.
- **Evaluación:** Todas las partidas se jugaron bajo las mismas condiciones para garantizar equidad.

---

*Reporte generado automáticamente por el sistema Chess-IA-2-Agent-Comparation*
"""
        
        return contenido