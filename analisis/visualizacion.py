"""
Módulo para generar visualizaciones de análisis de agentes de ajedrez.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Configurar matplotlib para evitar problemas con GUI
plt.switch_backend('Agg')
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class GeneradorVisualizaciones:
    """
    Genera gráficos y visualizaciones para análisis de agentes de ajedrez.
    """
    
    def __init__(self, estilo: str = 'seaborn-v0_8', paleta: str = 'husl'):
        """
        Inicializa el generador de visualizaciones.
        
        Args:
            estilo: Estilo de matplotlib
            paleta: Paleta de colores de seaborn
        """
        plt.style.use(estilo)
        sns.set_palette(paleta)
        self.figsize_default = (12, 8)
        self.figsize_small = (10, 6)
        self.figsize_large = (15, 10)
        
    def grafico_resultados_por_agente(self, metricas: Dict, archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico de barras con victorias, empates y derrotas por agente.
        
        Args:
            metricas: Diccionario con métricas básicas por agente
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico de resultados por agente...")
        
        # Preparar datos
        agentes = list(metricas.keys())
        victorias = [metricas[agente]['victorias'] for agente in agentes]
        empates = [metricas[agente]['empates'] for agente in agentes]
        derrotas = [metricas[agente]['derrotas'] for agente in agentes]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        x = np.arange(len(agentes))
        width = 0.25
        
        bars1 = ax.bar(x - width, victorias, width, label='Victorias', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x, empates, width, label='Empates', color='#FFD700', alpha=0.8)
        bars3 = ax.bar(x + width, derrotas, width, label='Derrotas', color='#DC143C', alpha=0.8)
        
        # Personalizar gráfico
        ax.set_xlabel('Agentes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Partidas', fontsize=12, fontweight='bold')
        ax.set_title('Resultados por Agente', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([agente.upper() for agente in agentes], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        def agregar_valores(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9, fontweight='bold')
        
        agregar_valores(bars1)
        agregar_valores(bars2)
        agregar_valores(bars3)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_tasas_rendimiento(self, metricas: Dict, archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico de barras apiladas con tasas de rendimiento.
        
        Args:
            metricas: Diccionario con métricas básicas por agente
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico de tasas de rendimiento...")
        
        # Preparar datos
        agentes = list(metricas.keys())
        tasa_victorias = [metricas[agente]['tasa_victorias'] * 100 for agente in agentes]
        tasa_empates = [metricas[agente]['tasa_empates'] * 100 for agente in agentes]
        tasa_derrotas = [metricas[agente]['tasa_derrotas'] * 100 for agente in agentes]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        bars1 = ax.bar(agentes, tasa_victorias, label='Victorias', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(agentes, tasa_empates, bottom=tasa_victorias, label='Empates', color='#FFD700', alpha=0.8)
        bars3 = ax.bar(agentes, tasa_derrotas, 
                      bottom=[i+j for i,j in zip(tasa_victorias, tasa_empates)], 
                      label='Derrotas', color='#DC143C', alpha=0.8)
        
        # Personalizar gráfico
        ax.set_xlabel('Agentes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
        ax.set_title('Tasas de Rendimiento por Agente', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels([agente.upper() for agente in agentes], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Agregar porcentajes en las barras
        for i, agente in enumerate(agentes):
            # Porcentaje de victorias
            if tasa_victorias[i] > 5:
                ax.text(i, tasa_victorias[i]/2, f'{tasa_victorias[i]:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
            
            # Porcentaje de empates
            if tasa_empates[i] > 5:
                ax.text(i, tasa_victorias[i] + tasa_empates[i]/2, f'{tasa_empates[i]:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='black')
            
            # Porcentaje de derrotas
            if tasa_derrotas[i] > 5:
                ax.text(i, tasa_victorias[i] + tasa_empates[i] + tasa_derrotas[i]/2, f'{tasa_derrotas[i]:.1f}%', 
                       ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_tiempos_promedio(self, metricas: Dict, archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico de barras con tiempos promedio por movimiento.
        
        Args:
            metricas: Diccionario con métricas básicas por agente
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico de tiempos promedio...")
        
        # Preparar datos
        agentes = list(metricas.keys())
        tiempos = [metricas[agente]['tiempo_promedio_por_movimiento'] for agente in agentes]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        bars = ax.bar(agentes, tiempos, color='skyblue', alpha=0.8, edgecolor='navy')
        
        # Personalizar gráfico
        ax.set_xlabel('Agentes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tiempo Promedio (segundos)', fontsize=12, fontweight='bold')
        ax.set_title('Tiempo Promedio por Movimiento', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels([agente.upper() for agente in agentes], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, tiempo in zip(bars, tiempos):
            height = bar.get_height()
            ax.annotate(f'{tiempo:.3f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_matriz_enfrentamientos(self, matriz: pd.DataFrame, archivo_salida: str = None) -> plt.Figure:
        """
        Crea un heatmap de la matriz de enfrentamientos.
        
        Args:
            matriz: DataFrame con matriz de enfrentamientos
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando matriz de enfrentamientos...")
        
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Crear heatmap
        sns.heatmap(matriz, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                   square=True, ax=ax, cbar_kws={'label': 'Porcentaje de Éxito (%)'})
        
        # Personalizar
        ax.set_title('Matriz de Enfrentamientos\n(% de éxito de fila vs columna)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Oponente', fontsize=12, fontweight='bold')
        ax.set_ylabel('Agente', fontsize=12, fontweight='bold')
        
        # Rotar etiquetas
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()], rotation=45, ha='right')
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()], rotation=0)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_evolucion_elo(self, df_resultados: pd.DataFrame, ratings_finales: Dict, 
                             archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico de evolución de ratings ELO.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            ratings_finales: Ratings ELO finales
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico de evolución ELO...")
        
        # Simular evolución de ELO (simplificado)
        agentes = list(ratings_finales.keys())
        fig, ax = plt.subplots(figsize=self.figsize_large)
        
        # Calcular evolución simplificada
        for agente in agentes:
            partidas_agente = df_resultados[
                (df_resultados['agente_blancas'] == agente) | 
                (df_resultados['agente_negras'] == agente)
            ].sort_values('numero_partida', na_position='last')
            
            if len(partidas_agente) > 0:
                # Simular evolución (desde rating inicial hasta final)
                rating_inicial = 1500
                rating_final = ratings_finales[agente]
                num_partidas = len(partidas_agente)
                
                # Evolución lineal simplificada (en realidad sería más compleja)
                evolucion = np.linspace(rating_inicial, rating_final, num_partidas + 1)
                
                ax.plot(range(num_partidas + 1), evolucion, marker='o', 
                       label=agente.upper(), linewidth=2, markersize=4)
        
        # Personalizar
        ax.set_xlabel('Número de Partidas', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rating ELO', fontsize=12, fontweight='bold')
        ax.set_title('Evolución de Ratings ELO', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_distribucion_movimientos(self, df_resultados: pd.DataFrame, 
                                        archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico de distribución de movimientos por partida.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico de distribución de movimientos...")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        fig.suptitle('Distribución de Movimientos por Partida', fontsize=16, fontweight='bold')
        
        # Obtener agentes únicos
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = list(agentes_blancas.union(agentes_negras))
        
        # Crear subplot para cada agente (máximo 4)
        for i, agente in enumerate(todos_agentes[:4]):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Filtrar partidas del agente
            partidas_agente = df_resultados[
                (df_resultados['agente_blancas'] == agente) | 
                (df_resultados['agente_negras'] == agente)
            ]
            
            movimientos = partidas_agente['movimientos_realizados']
            
            # Histograma
            ax.hist(movimientos, bins=15, alpha=0.7, edgecolor='black')
            ax.set_title(f'{agente.upper()}', fontweight='bold')
            ax.set_xlabel('Movimientos por Partida')
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            
            # Estadísticas
            media = movimientos.mean()
            mediana = movimientos.median()
            ax.axvline(media, color='red', linestyle='--', alpha=0.8, label=f'Media: {media:.1f}')
            ax.axvline(mediana, color='orange', linestyle='-', alpha=0.8, label=f'Mediana: {mediana:.1f}')
            ax.legend(fontsize=9)
        
        # Ocultar subplots vacíos
        for i in range(len(todos_agentes), 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def grafico_radar_metricas(self, metricas: Dict, archivo_salida: str = None) -> plt.Figure:
        """
        Crea un gráfico radar con múltiples métricas por agente.
        
        Args:
            metricas: Diccionario con métricas básicas por agente
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando gráfico radar de métricas...")
        
        # Preparar datos
        agentes = list(metricas.keys())
        categorias = ['Tasa Victorias', 'Consistencia Tiempo', 'Eficiencia Tiempo', 
                     'Duración Partidas', 'Performance General']
        
        # Normalizar métricas a escala 0-100
        datos_normalizados = {}
        
        for agente in agentes:
            m = metricas[agente]
            # Calcular tiempo máximo para normalización
            max_tiempo = max([metricas[a]['tiempo_promedio_por_movimiento'] for a in agentes])
            max_movimientos = max([metricas[a]['movimientos_promedio_por_partida'] for a in agentes])
            
            datos_normalizados[agente] = [
                m['tasa_victorias'] * 100,  # Tasa de victorias (ya en %)
                max(0, 100 - (m['tiempo_promedio_por_movimiento'] / max_tiempo * 100)),  # Consistencia tiempo (invertido)
                max(0, 100 - (m['tiempo_promedio_por_movimiento'] / max_tiempo * 100)),  # Eficiencia tiempo
                max(0, 100 - (m['movimientos_promedio_por_partida'] / max_movimientos * 100)),  # Duración (invertido)
                m['porcentaje_puntos']  # Performance general
            ]
        
        # Crear gráfico radar
        fig, ax = plt.subplots(figsize=self.figsize_default, subplot_kw=dict(projection='polar'))
        
        # Ángulos para cada categoría
        angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
        angulos += angulos[:1]  # Cerrar el círculo
        
        # Colores para cada agente
        colores = plt.cm.Set3(np.linspace(0, 1, len(agentes)))
        
        for i, agente in enumerate(agentes):
            valores = datos_normalizados[agente]
            valores += valores[:1]  # Cerrar el círculo
            
            ax.plot(angulos, valores, 'o-', linewidth=2, label=agente.upper(), color=colores[i])
            ax.fill(angulos, valores, alpha=0.25, color=colores[i])
        
        # Personalizar
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title('Perfil de Métricas por Agente', size=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado: {archivo_salida}")
        
        return fig
    
    def dashboard_completo(self, metricas: Dict, df_resultados: pd.DataFrame, 
                          matriz_enfrentamientos: pd.DataFrame, ratings_elo: Dict,
                          archivo_salida: str = None) -> plt.Figure:
        """
        Crea un dashboard completo con múltiples visualizaciones.
        
        Args:
            metricas: Métricas básicas por agente
            df_resultados: DataFrame con resultados
            matriz_enfrentamientos: Matriz de enfrentamientos
            ratings_elo: Ratings ELO finales
            archivo_salida: Ruta para guardar el gráfico
            
        Returns:
            Figure: Figura de matplotlib
        """
        logger.info("Generando dashboard completo...")
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Dashboard de Análisis de Agentes de Ajedrez', fontsize=20, fontweight='bold')
        
        # 1. Resultados por agente (2x3 grid, posición 1-2)
        ax1 = plt.subplot(3, 3, (1, 2))
        agentes = list(metricas.keys())
        victorias = [metricas[agente]['victorias'] for agente in agentes]
        empates = [metricas[agente]['empates'] for agente in agentes]
        derrotas = [metricas[agente]['derrotas'] for agente in agentes]
        
        x = np.arange(len(agentes))
        width = 0.25
        ax1.bar(x - width, victorias, width, label='Victorias', color='#2E8B57', alpha=0.8)
        ax1.bar(x, empates, width, label='Empates', color='#FFD700', alpha=0.8)
        ax1.bar(x + width, derrotas, width, label='Derrotas', color='#DC143C', alpha=0.8)
        ax1.set_title('Resultados por Agente', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.upper() for a in agentes], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Ratings ELO (posición 3)
        ax2 = plt.subplot(3, 3, 3)
        ratings = [ratings_elo[agente] for agente in agentes]
        bars = ax2.bar(agentes, ratings, color='lightcoral', alpha=0.8)
        ax2.set_title('Ratings ELO', fontweight='bold')
        ax2.set_xticklabels([a.upper() for a in agentes], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Agregar valores en barras
        for bar, rating in zip(bars, ratings):
            ax2.annotate(f'{rating:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 3. Tiempos promedio (posición 4)
        ax3 = plt.subplot(3, 3, 4)
        tiempos = [metricas[agente]['tiempo_promedio_por_movimiento'] for agente in agentes]
        ax3.bar(agentes, tiempos, color='skyblue', alpha=0.8)
        ax3.set_title('Tiempo Promedio/Mov', fontweight='bold')
        ax3.set_xticklabels([a.upper() for a in agentes], rotation=45, ha='right')
        ax3.set_ylabel('Segundos')
        ax3.grid(True, alpha=0.3)
        
        # 4. Matriz de enfrentamientos (posición 5-6)
        ax4 = plt.subplot(3, 3, (5, 6))
        sns.heatmap(matriz_enfrentamientos, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, square=True, ax=ax4, cbar=False)
        ax4.set_title('Matriz Enfrentamientos (%)', fontweight='bold')
        ax4.set_xticklabels([l.get_text().upper() for l in ax4.get_xticklabels()], rotation=45, ha='right')
        ax4.set_yticklabels([l.get_text().upper() for l in ax4.get_yticklabels()], rotation=0)
        
        # 5. Distribución de movimientos (posición 7)
        ax5 = plt.subplot(3, 3, 7)
        todos_movimientos = df_resultados['movimientos_realizados']
        ax5.hist(todos_movimientos, bins=15, alpha=0.7, edgecolor='black', color='lightgreen')
        ax5.set_title('Distribución Movimientos', fontweight='bold')
        ax5.set_xlabel('Movimientos por Partida')
        ax5.set_ylabel('Frecuencia')
        ax5.grid(True, alpha=0.3)
        
        # 6. Tasas de rendimiento (posición 8-9)
        ax6 = plt.subplot(3, 3, (8, 9))
        tasa_victorias = [metricas[agente]['tasa_victorias'] * 100 for agente in agentes]
        puntuaciones = [metricas[agente]['porcentaje_puntos'] for agente in agentes]
        
        x = np.arange(len(agentes))
        width = 0.35
        ax6.bar(x - width/2, tasa_victorias, width, label='% Victorias', color='green', alpha=0.7)
        ax6.bar(x + width/2, puntuaciones, width, label='% Puntos', color='blue', alpha=0.7)
        ax6.set_title('Rendimiento General', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([a.upper() for a in agentes], rotation=45, ha='right')
        ax6.set_ylabel('Porcentaje (%)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if archivo_salida:
            plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard guardado: {archivo_salida}")
        
        return fig
    
    def cerrar_todas_figuras(self):
        """Cierra todas las figuras abiertas para liberar memoria."""
        plt.close('all')


if __name__ == "__main__":
    # Ejemplo de uso
    print("Probando generador de visualizaciones...")
    
    # Datos de ejemplo
    metricas_ejemplo = {
        'minimax': {
            'victorias': 15, 'empates': 3, 'derrotas': 7,
            'tasa_victorias': 0.6, 'tasa_empates': 0.12, 'tasa_derrotas': 0.28,
            'tiempo_promedio_por_movimiento': 2.1, 'movimientos_promedio_por_partida': 45,
            'porcentaje_puntos': 66.0
        },
        'mcts': {
            'victorias': 12, 'empates': 5, 'derrotas': 8,
            'tasa_victorias': 0.48, 'tasa_empates': 0.2, 'tasa_derrotas': 0.32,
            'tiempo_promedio_por_movimiento': 3.2, 'movimientos_promedio_por_partida': 52,
            'porcentaje_puntos': 58.0
        }
    }
    
    # Crear visualizador
    viz = GeneradorVisualizaciones()
    
    # Generar un gráfico de ejemplo
    fig = viz.grafico_resultados_por_agente(metricas_ejemplo, 'test_resultados.png')
    
    print("Visualización de prueba generada: test_resultados.png")
    print("Prueba completada")
