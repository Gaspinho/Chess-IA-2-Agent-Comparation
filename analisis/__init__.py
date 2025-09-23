"""
Módulo de análisis - Métricas, visualizaciones y reportes.
"""

from .metricas_desempeno import CalculadorMetricas, generar_tabla_comparativa
from .visualizacion import GeneradorVisualizaciones
from .generar_reporte import GeneradorReporte

__all__ = [
    'CalculadorMetricas',
    'generar_tabla_comparativa',
    'GeneradorVisualizaciones', 
    'GeneradorReporte'
]