"""
Módulo para crear visualizaciones de resultados.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os

plt.style.use('default')
sns.set_palette("husl")


class GeneradorVisualizaciones:
    """Genera gráficos y visualizaciones de los resultados."""
    
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos
        self.agentes = self._obtener_agentes()
    
    def _obtener_agentes(self) -> List[str]:
        """Obtiene lista de agentes únicos."""
        agentes = set()
        agentes.update(self.datos['agente_blancas'].unique())
        agentes.update(self.datos['agente_negras'].unique())
        return sorted(list(agentes))
    
    def grafico_victorias_por_agente(self, guardar_como: str = None) -> plt.Figure:
        """Crea gráfico de barras con victorias por agente."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        victorias = {}
        for agente in self.agentes:
            como_blancas = self.datos[self.datos['agente_blancas'] == agente]
            como_negras = self.datos[self.datos['agente_negras'] == agente]
            
            victorias_blancas = (como_blancas['ganador'] == agente).sum()
            victorias_negras = (como_negras['ganador'] == agente).sum()
            victorias[agente] = victorias_blancas + victorias_negras
        
        agentes_ordenados = sorted(victorias.keys(), key=lambda x: victorias[x], reverse=True)
        valores = [victorias[agente] for agente in agentes_ordenados]
        
        bars = ax.bar(agentes_ordenados, valores, color=sns.color_palette("husl", len(agentes_ordenados)))
        ax.set_title('Total de Victorias por Agente', fontsize=16, fontweight='bold')
        ax.set_xlabel('Agente', fontsize=12)
        ax.set_ylabel('Número de Victorias', fontsize=12)
        
        # Agregar valores en las barras
        for bar, valor in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(valor), ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    def grafico_matriz_enfrentamientos(self, guardar_como: str = None) -> plt.Figure:
        """Crea matriz de calor de enfrentamientos."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Crear matriz de resultados
        matriz = pd.DataFrame(0.0, index=self.agentes, columns=self.agentes)
        
        for _, fila in self.datos.iterrows():
            agente_a = fila['agente_blancas']
            agente_b = fila['agente_negras']
            ganador = fila['ganador']
            
            if pd.notna(ganador):
                if ganador == agente_a:
                    matriz.loc[agente_a, agente_b] += 1
                else:
                    matriz.loc[agente_b, agente_a] += 1
        
        # Convertir a porcentajes
        for i in self.agentes:
            for j in self.agentes:
                if i != j:
                    total = matriz.loc[i, j] + matriz.loc[j, i]
                    if total > 0:
                        matriz.loc[i, j] = matriz.loc[i, j] / total * 100
        
        sns.heatmap(matriz, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   center=50, ax=ax, square=True)
        ax.set_title('Matriz de Enfrentamientos (% Victorias)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Oponente', fontsize=12)
        ax.set_ylabel('Agente', fontsize=12)
        
        plt.tight_layout()
        
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    def grafico_tiempo_por_jugada(self, guardar_como: str = None) -> plt.Figure:
        """Crea gráfico de tiempo promedio por jugada."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tiempos = {}
        for agente in self.agentes:
            como_blancas = self.datos[self.datos['agente_blancas'] == agente]
            como_negras = self.datos[self.datos['agente_negras'] == agente]
            
            tiempo_blancas = (como_blancas['tiempo_blancas'] / (como_blancas['num_jugadas'] / 2)).mean()
            tiempo_negras = (como_negras['tiempo_negras'] / (como_negras['num_jugadas'] / 2)).mean()
            
            tiempos[agente] = np.nanmean([tiempo_blancas, tiempo_negras])
        
        agentes_ordenados = sorted(tiempos.keys(), key=lambda x: tiempos[x])
        valores = [tiempos[agente] for agente in agentes_ordenados]
        
        bars = ax.bar(agentes_ordenados, valores, color=sns.color_palette("viridis", len(agentes_ordenados)))
        ax.set_title('Tiempo Promedio por Jugada', fontsize=16, fontweight='bold')
        ax.set_xlabel('Agente', fontsize=12)
        ax.set_ylabel('Tiempo (segundos)', fontsize=12)
        
        # Agregar valores en las barras
        for bar, valor in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generar_todas_visualizaciones(self, directorio_salida: str = "resultados/"):
        """Genera todas las visualizaciones y las guarda."""
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Generar gráficos
        self.grafico_victorias_por_agente(os.path.join(directorio_salida, "victorias_por_agente.png"))
        self.grafico_matriz_enfrentamientos(os.path.join(directorio_salida, "matriz_enfrentamientos.png"))
        self.grafico_tiempo_por_jugada(os.path.join(directorio_salida, "tiempo_por_jugada.png"))
        
        plt.close('all')  # Cerrar todas las figuras para liberar memoria