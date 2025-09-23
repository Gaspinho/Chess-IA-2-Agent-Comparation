"""
Módulo para calcular métricas de desempeño de agentes de ajedrez.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CalculadorMetricas:
    """Calcula métricas de desempeño de agentes."""
    
    def __init__(self, datos_resultados: pd.DataFrame):
        """
        Inicializa con los datos de resultados.
        
        Args:
            datos_resultados: DataFrame con resultados de partidas
        """
        self.datos = datos_resultados
        self.agentes = self._obtener_agentes_unicos()
    
    def _obtener_agentes_unicos(self) -> List[str]:
        """Obtiene lista de agentes únicos."""
        agentes = set()
        agentes.update(self.datos['agente_blancas'].unique())
        agentes.update(self.datos['agente_negras'].unique())
        return list(agentes)
    
    def calcular_metricas_generales(self) -> Dict[str, Any]:
        """Calcula métricas generales del dataset."""
        return {
            'total_partidas': len(self.datos),
            'agentes_evaluados': len(self.agentes),
            'partidas_por_agente': len(self.datos) // len(self.agentes) if self.agentes else 0,
            'tiempo_promedio_partida': self.datos['tiempo_total'].mean(),
            'jugadas_promedio': self.datos['num_jugadas'].mean(),
            'tasa_completacion': (self.datos['tipo_fin'] != 'error').mean()
        }
    
    def calcular_metricas_por_agente(self) -> Dict[str, Dict[str, float]]:
        """Calcula métricas individuales por agente."""
        metricas = {}
        
        for agente in self.agentes:
            # Filtrar partidas del agente
            como_blancas = self.datos[self.datos['agente_blancas'] == agente]
            como_negras = self.datos[self.datos['agente_negras'] == agente]
            
            # Calcular victorias
            victorias_blancas = (como_blancas['ganador'] == agente).sum()
            victorias_negras = (como_negras['ganador'] == agente).sum()
            total_victorias = victorias_blancas + victorias_negras
            
            # Calcular empates
            empates_blancas = (como_blancas['ganador'].isna() & (como_blancas['tipo_fin'] == 'empate')).sum()
            empates_negras = (como_negras['ganador'].isna() & (como_negras['tipo_fin'] == 'empate')).sum()
            total_empates = empates_blancas + empates_negras
            
            # Total partidas
            total_partidas = len(como_blancas) + len(como_negras)
            
            # Tiempos promedio
            tiempo_blancas = como_blancas['tiempo_blancas'].mean()
            tiempo_negras = como_negras['tiempo_negras'].mean()
            tiempo_promedio = np.nanmean([tiempo_blancas, tiempo_negras])
            
            # Recompensas promedio
            recompensa_blancas = como_blancas['recompensa_blancas'].mean()
            recompensa_negras = como_negras['recompensa_negras'].mean()
            recompensa_promedio = np.nanmean([recompensa_blancas, recompensa_negras])
            
            metricas[agente] = {
                'total_partidas': total_partidas,
                'victorias': total_victorias,
                'empates': total_empates,
                'derrotas': total_partidas - total_victorias - total_empates,
                'tasa_victoria': total_victorias / total_partidas if total_partidas > 0 else 0,
                'tasa_empate': total_empates / total_partidas if total_partidas > 0 else 0,
                'tiempo_promedio_jugada': tiempo_promedio / 2 if not np.isnan(tiempo_promedio) else 0,
                'recompensa_promedio': recompensa_promedio if not np.isnan(recompensa_promedio) else 0
            }
        
        return metricas


def generar_tabla_comparativa(metricas: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Genera tabla comparativa de métricas."""
    filas = []
    for agente, stats in metricas.items():
        filas.append({
            'Agente': agente,
            'Partidas': stats['total_partidas'],
            'Victorias': stats['victorias'],
            'Empates': stats['empates'],
            'Derrotas': stats['derrotas'],
            'Tasa Victoria (%)': stats['tasa_victoria'] * 100,
            'Tasa Empate (%)': stats['tasa_empate'] * 100,
            'Tiempo/Jugada (s)': stats['tiempo_promedio_jugada'],
            'Recompensa Promedio': stats['recompensa_promedio']
        })
    
    return pd.DataFrame(filas).round(2)