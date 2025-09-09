"""
Generador de reportes en Markdown para análisis de agentes de ajedrez.
"""

import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class GeneradorReportes:
    """
    Genera reportes detallados en formato Markdown.
    """
    
    def __init__(self):
        """Inicializa el generador de reportes."""
        self.titulo_principal = "Análisis Comparativo de Agentes de Ajedrez"
        
    def generar_reporte_completo(self, metricas_basicas: Dict, metricas_avanzadas: Dict,
                                df_resultados: pd.DataFrame, matriz_enfrentamientos: pd.DataFrame,
                                ratings_elo: Dict, archivos_graficos: Dict,
                                archivo_salida: str = "analisis/reporte.md") -> str:
        """
        Genera un reporte completo en Markdown.
        
        Args:
            metricas_basicas: Métricas básicas por agente
            metricas_avanzadas: Métricas avanzadas por agente
            df_resultados: DataFrame con resultados de partidas
            matriz_enfrentamientos: Matriz de enfrentamientos
            ratings_elo: Ratings ELO finales
            archivos_graficos: Diccionario con rutas de gráficos generados
            archivo_salida: Ruta del archivo de salida
            
        Returns:
            str: Contenido del reporte en Markdown
        """
        logger.info("Generando reporte completo...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        
        contenido = []
        
        # Encabezado
        contenido.extend(self._generar_encabezado())
        
        # Resumen ejecutivo
        contenido.extend(self._generar_resumen_ejecutivo(metricas_basicas, ratings_elo))
        
        # Ranking general
        contenido.extend(self._generar_ranking_general(metricas_basicas, ratings_elo))
        
        # Análisis detallado por agente
        contenido.extend(self._generar_analisis_por_agente(metricas_basicas, metricas_avanzadas))
        
        # Análisis de enfrentamientos
        contenido.extend(self._generar_analisis_enfrentamientos(matriz_enfrentamientos))
        
        # Métricas de rendimiento
        contenido.extend(self._generar_metricas_rendimiento(df_resultados))
        
        # Gráficos y visualizaciones
        contenido.extend(self._generar_seccion_graficos(archivos_graficos))
        
        # Conclusiones y recomendaciones
        contenido.extend(self._generar_conclusiones(metricas_basicas, ratings_elo))
        
        # Detalles técnicos
        contenido.extend(self._generar_detalles_tecnicos(df_resultados))
        
        # Escribir archivo
        contenido_completo = "\n".join(contenido)
        
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write(contenido_completo)
        
        logger.info(f"Reporte generado: {archivo_salida}")
        return contenido_completo
    
    def _generar_encabezado(self) -> List[str]:
        """Genera el encabezado del reporte."""
        return [
            f"# {self.titulo_principal}",
            "",
            f"**Fecha de generación:** {datetime.now().strftime('%d de %B de %Y, %H:%M:%S')}",
            "",
            "## Índice",
            "",
            "1. [Resumen Ejecutivo](#resumen-ejecutivo)",
            "2. [Ranking General](#ranking-general)",
            "3. [Análisis por Agente](#análisis-por-agente)",
            "4. [Análisis de Enfrentamientos](#análisis-de-enfrentamientos)",
            "5. [Métricas de Rendimiento](#métricas-de-rendimiento)",
            "6. [Visualizaciones](#visualizaciones)",
            "7. [Conclusiones](#conclusiones)",
            "8. [Detalles Técnicos](#detalles-técnicos)",
            "",
            "---",
            ""
        ]
    
    def _generar_resumen_ejecutivo(self, metricas: Dict, ratings_elo: Dict) -> List[str]:
        """Genera el resumen ejecutivo."""
        # Encontrar el mejor agente
        mejor_agente = max(ratings_elo.items(), key=lambda x: x[1])
        agentes_ordenados = sorted(ratings_elo.items(), key=lambda x: x[1], reverse=True)
        
        # Estadísticas generales
        total_partidas = sum([metricas[agente]['total_partidas'] for agente in metricas]) // 2  # Dividir por 2 porque cada partida se cuenta dos veces
        total_agentes = len(metricas)
        
        return [
            "## Resumen Ejecutivo",
            "",
            f"Este reporte presenta un análisis comparativo de **{total_agentes} agentes de ajedrez** "
            f"basado en **{total_partidas} partidas** jugadas entre todos los agentes.",
            "",
            "### Principales Hallazgos",
            "",
            f"🏆 **Mejor Agente:** {mejor_agente[0].upper()} (Rating ELO: {mejor_agente[1]:.0f})",
            "",
            "#### Ranking Final:",
            ""
        ] + [
            f"{i+1}. **{agente.upper()}** - Rating ELO: {rating:.0f} "
            f"({metricas[agente]['porcentaje_puntos']:.1f}% puntos)"
            for i, (agente, rating) in enumerate(agentes_ordenados)
        ] + [
            "",
            "### Tipos de Agentes Evaluados",
            "",
            "- **Agentes de Búsqueda:** Minimax y MCTS",
            "- **Agentes de Aprendizaje por Refuerzo:** DQN y PPO",
            "",
            "---",
            ""
        ]
    
    def _generar_ranking_general(self, metricas: Dict, ratings_elo: Dict) -> List[str]:
        """Genera la tabla de ranking general."""
        agentes_ordenados = sorted(ratings_elo.items(), key=lambda x: x[1], reverse=True)
        
        contenido = [
            "## Ranking General",
            "",
            "| Posición | Agente | Rating ELO | Puntos | Victorias | Empates | Derrotas | % Victorias | Tiempo Promedio |",
            "|----------|--------|------------|--------|-----------|---------|----------|-------------|-----------------|"
        ]
        
        for i, (agente, rating) in enumerate(agentes_ordenados):
            m = metricas[agente]
            contenido.append(
                f"| {i+1} | **{agente.upper()}** | {rating:.0f} | "
                f"{m['puntos']:.1f}/{m['total_partidas']} | "
                f"{m['victorias']} | {m['empates']} | {m['derrotas']} | "
                f"{m['tasa_victorias']*100:.1f}% | "
                f"{m['tiempo_promedio_por_movimiento']:.3f}s |"
            )
        
        contenido.extend([
            "",
            "### Explicación de Métricas",
            "",
            "- **Rating ELO:** Sistema de clasificación que considera la dificultad del oponente",
            "- **Puntos:** Victorias + 0.5 × Empates",
            "- **% Victorias:** Porcentaje de partidas ganadas",
            "- **Tiempo Promedio:** Tiempo promedio por movimiento en segundos",
            "",
            "---",
            ""
        ])
        
        return contenido
    
    def _generar_analisis_por_agente(self, metricas_basicas: Dict, metricas_avanzadas: Dict) -> List[str]:
        """Genera análisis detallado por agente."""
        contenido = [
            "## Análisis por Agente",
            ""
        ]
        
        for agente in sorted(metricas_basicas.keys()):
            basicas = metricas_basicas[agente]
            avanzadas = metricas_avanzadas[agente]
            
            contenido.extend([
                f"### {agente.upper()}",
                "",
                "#### Estadísticas Básicas",
                "",
                f"- **Partidas jugadas:** {basicas['total_partidas']}",
                f"- **Record:** {basicas['victorias']}V - {basicas['empates']}E - {basicas['derrotas']}D",
                f"- **Puntuación:** {basicas['puntos']:.1f} puntos ({basicas['porcentaje_puntos']:.1f}%)",
                f"- **Tiempo promedio por movimiento:** {basicas['tiempo_promedio_por_movimiento']:.3f}s",
                "",
                "#### Rendimiento por Color",
                "",
                f"- **Como blancas:** {basicas['victorias_como_blancas']} victorias de {basicas['partidas_como_blancas']} partidas "
                f"({basicas['eficiencia_blancas']*100:.1f}% eficiencia)",
                f"- **Como negras:** {basicas['victorias_como_negras']} victorias de {basicas['partidas_como_negras']} partidas "
                f"({basicas['eficiencia_negras']*100:.1f}% eficiencia)",
                "",
                "#### Métricas Avanzadas",
                "",
                f"- **Consistencia temporal:** {avanzadas['consistencia_temporal']:.3f}",
                f"- **Tasa de finalización decisiva:** {avanzadas['tasa_finalizacion_decisiva']*100:.1f}%",
                f"- **Movimientos promedio por victoria:** {avanzadas.get('movimientos_promedio_victoria', 'N/A'):.1f}",
                f"- **Oponentes enfrentados:** {avanzadas['oponentes_enfrentados']}",
                "",
                "#### Formas de Terminación",
                ""
            ])
            
            # Agregar terminaciones
            terminaciones = avanzadas['terminaciones_por_tipo']
            for tipo, cantidad in terminaciones.items():
                contenido.append(f"- **{tipo.replace('_', ' ').title()}:** {cantidad} partidas")
            
            contenido.extend([
                "",
                "---",
                ""
            ])
        
        return contenido
    
    def _generar_analisis_enfrentamientos(self, matriz: pd.DataFrame) -> List[str]:
        """Genera análisis de enfrentamientos."""
        contenido = [
            "## Análisis de Enfrentamientos",
            "",
            "### Matriz de Enfrentamientos",
            "",
            "La siguiente tabla muestra el porcentaje de éxito de cada agente (fila) contra cada oponente (columna):",
            "",
            "| Agente \\ Oponente |"
        ]
        
        # Encabezado de la tabla
        agentes = list(matriz.index)
        contenido[5] += " | ".join([f" **{agente.upper()}** " for agente in agentes]) + " |"
        contenido.append("|" + "---|" * (len(agentes) + 1))
        
        # Filas de la tabla
        for agente_fila in agentes:
            fila = f"| **{agente_fila.upper()}** |"
            for agente_col in agentes:
                if agente_fila == agente_col:
                    fila += " - |"
                else:
                    valor = matriz.loc[agente_fila, agente_col]
                    if pd.isna(valor):
                        fila += " N/A |"
                    else:
                        fila += f" {valor:.1f}% |"
            contenido.append(fila)
        
        # Análisis de dominancia
        contenido.extend([
            "",
            "### Análisis de Dominancia",
            ""
        ])
        
        # Encontrar mejor y peor enfrentamiento para cada agente
        for agente in agentes:
            mejores = []
            peores = []
            
            for oponente in agentes:
                if agente != oponente:
                    valor = matriz.loc[agente, oponente]
                    if not pd.isna(valor):
                        if valor >= 60:
                            mejores.append(f"{oponente.upper()} ({valor:.1f}%)")
                        elif valor <= 40:
                            peores.append(f"{oponente.upper()} ({valor:.1f}%)")
            
            contenido.append(f"**{agente.upper()}:**")
            if mejores:
                contenido.append(f"- ✅ Domina contra: {', '.join(mejores)}")
            if peores:
                contenido.append(f"- ❌ Lucha contra: {', '.join(peores)}")
            contenido.append("")
        
        contenido.extend([
            "---",
            ""
        ])
        
        return contenido
    
    def _generar_metricas_rendimiento(self, df_resultados: pd.DataFrame) -> List[str]:
        """Genera sección de métricas de rendimiento."""
        contenido = [
            "## Métricas de Rendimiento",
            "",
            "### Estadísticas Generales de las Partidas",
            ""
        ]
        
        # Estadísticas generales
        total_partidas = len(df_resultados)
        movimientos_promedio = df_resultados['movimientos_realizados'].mean()
        movimientos_std = df_resultados['movimientos_realizados'].std()
        tiempo_promedio_partida = df_resultados['tiempo_total_partida'].mean()
        
        contenido.extend([
            f"- **Total de partidas analizadas:** {total_partidas}",
            f"- **Movimientos promedio por partida:** {movimientos_promedio:.1f} ± {movimientos_std:.1f}",
            f"- **Duración promedio por partida:** {tiempo_promedio_partida/60:.1f} minutos",
            "",
            "### Distribución de Resultados",
            ""
        ])
        
        # Distribución de resultados
        resultados = df_resultados['ganador'].value_counts()
        total = len(df_resultados)
        
        for resultado, cantidad in resultados.items():
            porcentaje = (cantidad / total) * 100
            contenido.append(f"- **{resultado.title()}:** {cantidad} partidas ({porcentaje:.1f}%)")
        
        contenido.extend([
            "",
            "### Formas de Terminación",
            ""
        ])
        
        # Formas de terminación
        terminaciones = df_resultados['razon_fin'].value_counts()
        
        for terminacion, cantidad in terminaciones.items():
            porcentaje = (cantidad / total) * 100
            contenido.append(f"- **{terminacion.replace('_', ' ').title()}:** {cantidad} partidas ({porcentaje:.1f}%)")
        
        contenido.extend([
            "",
            "---",
            ""
        ])
        
        return contenido
    
    def _generar_seccion_graficos(self, archivos_graficos: Dict) -> List[str]:
        """Genera sección de gráficos y visualizaciones."""
        contenido = [
            "## Visualizaciones",
            "",
            "Las siguientes visualizaciones resumen los principales hallazgos del análisis:",
            ""
        ]
        
        # Agregar cada gráfico si existe el archivo
        graficos_descripciones = {
            'resultados': ('Resultados por Agente', 'Distribución de victorias, empates y derrotas por agente'),
            'tasas_rendimiento': ('Tasas de Rendimiento', 'Porcentajes de victorias, empates y derrotas'),
            'tiempos': ('Tiempos Promedio', 'Tiempo promedio por movimiento de cada agente'),
            'matriz': ('Matriz de Enfrentamientos', 'Heatmap de éxito de cada agente contra sus oponentes'),
            'elo': ('Evolución de Ratings ELO', 'Progresión de los ratings durante las partidas'),
            'movimientos': ('Distribución de Movimientos', 'Histograma de movimientos por partida'),
            'radar': ('Perfil de Métricas', 'Gráfico radar con múltiples métricas normalizadas'),
            'dashboard': ('Dashboard Completo', 'Resumen visual de todas las métricas principales')
        }
        
        for clave, (titulo, descripcion) in graficos_descripciones.items():
            if clave in archivos_graficos and os.path.exists(archivos_graficos[clave]):
                contenido.extend([
                    f"### {titulo}",
                    "",
                    descripcion,
                    "",
                    f"![{titulo}]({archivos_graficos[clave]})",
                    ""
                ])
        
        contenido.extend([
            "---",
            ""
        ])
        
        return contenido
    
    def _generar_conclusiones(self, metricas: Dict, ratings_elo: Dict) -> List[str]:
        """Genera conclusiones y recomendaciones."""
        agentes_ordenados = sorted(ratings_elo.items(), key=lambda x: x[1], reverse=True)
        mejor_agente = agentes_ordenados[0][0]
        peor_agente = agentes_ordenados[-1][0]
        
        # Análisis de tiempos
        agente_mas_rapido = min(metricas.items(), key=lambda x: x[1]['tiempo_promedio_por_movimiento'])
        agente_mas_lento = max(metricas.items(), key=lambda x: x[1]['tiempo_promedio_por_movimiento'])
        
        contenido = [
            "## Conclusiones",
            "",
            "### Principales Hallazgos",
            "",
            f"1. **Mejor Rendimiento General:** {mejor_agente.upper()} demostró el mejor rendimiento "
            f"con un rating ELO de {ratings_elo[mejor_agente]:.0f} puntos.",
            "",
            f"2. **Eficiencia Temporal:** {agente_mas_rapido[0].upper()} fue el más rápido con "
            f"{agente_mas_rapido[1]['tiempo_promedio_por_movimiento']:.3f}s por movimiento, mientras que "
            f"{agente_mas_lento[0].upper()} fue el más lento con "
            f"{agente_mas_lento[1]['tiempo_promedio_por_movimiento']:.3f}s.",
            "",
            "3. **Consistencia:** Los agentes de búsqueda (Minimax/MCTS) tienden a ser más predecibles "
            "en sus tiempos de respuesta comparados con los agentes de RL.",
            "",
            "### Recomendaciones",
            "",
            "#### Para Mejoras Futuras:",
            "",
            "- **Agentes de Búsqueda:** Considerar aumentar la profundidad de búsqueda o mejorar "
            "las funciones de evaluación",
            "- **Agentes RL:** Más tiempo de entrenamiento y ajuste de hiperparámetros",
            "- **General:** Implementar técnicas de apertura y finales específicas",
            "",
            "#### Casos de Uso Recomendados:",
            ""
        ]
        
        # Recomendaciones específicas por agente
        for agente, rating in agentes_ordenados:
            m = metricas[agente]
            if rating > 1600:
                nivel = "Competitivo"
            elif rating > 1400:
                nivel = "Intermedio"
            else:
                nivel = "Principiante"
            
            contenido.append(f"- **{agente.upper()}:** {nivel} - "
                           f"Tiempo promedio {m['tiempo_promedio_por_movimiento']:.3f}s")
        
        contenido.extend([
            "",
            "---",
            ""
        ])
        
        return contenido
    
    def _generar_detalles_tecnicos(self, df_resultados: pd.DataFrame) -> List[str]:
        """Genera sección de detalles técnicos."""
        return [
            "## Detalles Técnicos",
            "",
            "### Metodología de Evaluación",
            "",
            "- **Sistema de Puntuación:** Victorias = 1 punto, Empates = 0.5 puntos, Derrotas = 0 puntos",
            "- **Rating ELO:** Sistema dinámico que considera la fuerza del oponente (K=32)",
            "- **Tiempo Límite:** 5 segundos por movimiento (configurable)",
            "- **Formato de Torneo:** Todos contra todos con alternancia de colores",
            "",
            "### Configuración de Agentes",
            "",
            "#### Minimax",
            "- Profundidad de búsqueda: 3 niveles",
            "- Poda alfa-beta activada",
            "- Función de evaluación: Material + posición + movilidad",
            "",
            "#### MCTS",
            "- Simulaciones por movimiento: 1000",
            "- Parámetro de exploración (C): 1.4",
            "- Política de simulación: Priorización de capturas y jaques",
            "",
            "#### DQN",
            "- Arquitectura: Red neuronal convolucional",
            "- Buffer de experiencia: 50,000 transiciones",
            "- Tasa de aprendizaje: 1e-4",
            "",
            "#### PPO",
            "- Pasos por actualización: 2048",
            "- Épocas por actualización: 10",
            "- Tasa de aprendizaje: 3e-4",
            "",
            "### Limitaciones del Estudio",
            "",
            "- Número limitado de partidas por enfrentamiento",
            "- Agentes RL pueden requerir más entrenamiento",
            "- Tiempo límite puede favorecer a agentes más rápidos",
            "- No se consideraron aperturas específicas",
            "",
            "### Información del Sistema",
            "",
            f"- **Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Total de partidas:** {len(df_resultados)}",
            f"- **Duración total del análisis:** {df_resultados['tiempo_total_partida'].sum()/3600:.2f} horas",
            "",
            "---",
            "",
            "*Reporte generado automáticamente por el Sistema de Análisis de Agentes de Ajedrez*"
        ]
    
    def generar_reporte_rapido(self, metricas: Dict, ratings_elo: Dict, 
                              archivo_salida: str = "analisis/reporte_rapido.md") -> str:
        """
        Genera un reporte rápido con información esencial.
        
        Args:
            metricas: Métricas básicas por agente
            ratings_elo: Ratings ELO finales
            archivo_salida: Ruta del archivo de salida
            
        Returns:
            str: Contenido del reporte
        """
        logger.info("Generando reporte rápido...")
        
        agentes_ordenados = sorted(ratings_elo.items(), key=lambda x: x[1], reverse=True)
        
        contenido = [
            "# Reporte Rápido - Análisis de Agentes de Ajedrez",
            "",
            f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Ranking Final",
            "",
            "| Pos | Agente | ELO | Puntos | V-E-D | % Victorias |",
            "|-----|--------|-----|--------|-------|-------------|"
        ]
        
        for i, (agente, rating) in enumerate(agentes_ordenados):
            m = metricas[agente]
            contenido.append(
                f"| {i+1} | **{agente.upper()}** | {rating:.0f} | "
                f"{m['puntos']:.1f} | {m['victorias']}-{m['empates']}-{m['derrotas']} | "
                f"{m['tasa_victorias']*100:.1f}% |"
            )
        
        contenido.extend([
            "",
            "## Resumen",
            "",
            f"🏆 **Campeón:** {agentes_ordenados[0][0].upper()}",
            f"⚡ **Más Rápido:** {min(metricas.items(), key=lambda x: x[1]['tiempo_promedio_por_movimiento'])[0].upper()}",
            f"🎯 **Más Efectivo:** {max(metricas.items(), key=lambda x: x[1]['tasa_victorias'])[0].upper()}",
            ""
        ])
        
        # Escribir archivo
        contenido_completo = "\n".join(contenido)
        
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write(contenido_completo)
        
        logger.info(f"Reporte rápido generado: {archivo_salida}")
        return contenido_completo


if __name__ == "__main__":
    # Ejemplo de uso
    print("Probando generador de reportes...")
    
    # Datos de ejemplo
    metricas_ejemplo = {
        'minimax': {
            'total_partidas': 20, 'victorias': 12, 'empates': 4, 'derrotas': 4,
            'puntos': 14.0, 'porcentaje_puntos': 70.0, 'tasa_victorias': 0.6,
            'tiempo_promedio_por_movimiento': 2.1, 'victorias_como_blancas': 6,
            'victorias_como_negras': 6, 'partidas_como_blancas': 10,
            'partidas_como_negras': 10, 'eficiencia_blancas': 0.7, 'eficiencia_negras': 0.7
        }
    }
    
    ratings_ejemplo = {'minimax': 1650}
    
    # Generar reporte rápido
    generador = GeneradorReportes()
    reporte = generador.generar_reporte_rapido(metricas_ejemplo, ratings_ejemplo, "test_reporte.md")
    
    print("Reporte de prueba generado: test_reporte.md")
    print("Contenido:")
    print(reporte[:500] + "...")
    print("Prueba completada")
