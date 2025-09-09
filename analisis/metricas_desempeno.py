"""
Módulo para calcular métricas de desempeño de agentes de ajedrez.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
import chess

logger = logging.getLogger(__name__)


class CalculadorMetricas:
    """
    Calcula diversas métricas de desempeño para agentes de ajedrez.
    """
    
    def __init__(self):
        """Inicializa el calculador de métricas."""
        self.metricas_calculadas = {}
    
    def calcular_metricas_basicas(self, df_resultados: pd.DataFrame) -> Dict:
        """
        Calcula métricas básicas de desempeño.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            
        Returns:
            dict: Métricas básicas por agente
        """
        logger.info("Calculando métricas básicas...")
        
        metricas = {}
        
        # Obtener lista de agentes
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = list(agentes_blancas.union(agentes_negras))
        
        for agente in todos_agentes:
            # Filtrar partidas del agente
            partidas_blancas = df_resultados[df_resultados['agente_blancas'] == agente]
            partidas_negras = df_resultados[df_resultados['agente_negras'] == agente]
            
            # Conteos básicos
            victorias_blancas = len(partidas_blancas[partidas_blancas['ganador'] == 'blancas'])
            empates_blancas = len(partidas_blancas[partidas_blancas['ganador'] == 'empate'])
            derrotas_blancas = len(partidas_blancas[partidas_blancas['ganador'] == 'negras'])
            
            victorias_negras = len(partidas_negras[partidas_negras['ganador'] == 'negras'])
            empates_negras = len(partidas_negras[partidas_negras['ganador'] == 'empate'])
            derrotas_negras = len(partidas_negras[partidas_negras['ganador'] == 'blancas'])
            
            total_victorias = victorias_blancas + victorias_negras
            total_empates = empates_blancas + empates_negras
            total_derrotas = derrotas_blancas + derrotas_negras
            total_partidas = total_victorias + total_empates + total_derrotas
            
            # Métricas básicas
            if total_partidas > 0:
                tasa_victorias = total_victorias / total_partidas
                tasa_empates = total_empates / total_partidas
                tasa_derrotas = total_derrotas / total_partidas
                puntuacion_elo_aproximada = self._calcular_elo_aproximado(
                    total_victorias, total_empates, total_derrotas
                )
            else:
                tasa_victorias = tasa_empates = tasa_derrotas = 0
                puntuacion_elo_aproximada = 1500  # ELO base
            
            # Métricas de tiempo
            todas_partidas = pd.concat([partidas_blancas, partidas_negras])
            tiempo_promedio_blancas = partidas_blancas['tiempo_promedio_blancas'].mean()
            tiempo_promedio_negras = partidas_negras['tiempo_promedio_negras'].mean()
            
            tiempo_promedio = np.nanmean([tiempo_promedio_blancas, tiempo_promedio_negras])
            tiempo_max_blancas = partidas_blancas['tiempo_max_blancas'].max()
            tiempo_max_negras = partidas_negras['tiempo_max_negras'].max()
            tiempo_max = np.nanmax([tiempo_max_blancas, tiempo_max_negras])
            
            # Métricas de duración de partidas
            movimientos_promedio = todas_partidas['movimientos_realizados'].mean()
            movimientos_std = todas_partidas['movimientos_realizados'].std()
            
            metricas[agente] = {
                # Resultados básicos
                'total_partidas': total_partidas,
                'victorias': total_victorias,
                'empates': total_empates,
                'derrotas': total_derrotas,
                
                # Tasas
                'tasa_victorias': tasa_victorias,
                'tasa_empates': tasa_empates,
                'tasa_derrotas': tasa_derrotas,
                
                # Puntuación
                'puntos': total_victorias + (total_empates * 0.5),
                'porcentaje_puntos': ((total_victorias + (total_empates * 0.5)) / total_partidas * 100) if total_partidas > 0 else 0,
                'elo_aproximado': puntuacion_elo_aproximada,
                
                # Tiempos
                'tiempo_promedio_por_movimiento': tiempo_promedio,
                'tiempo_maximo_por_movimiento': tiempo_max,
                
                # Duración de partidas
                'movimientos_promedio_por_partida': movimientos_promedio,
                'movimientos_std': movimientos_std,
                
                # Performance por color
                'victorias_como_blancas': victorias_blancas,
                'victorias_como_negras': victorias_negras,
                'partidas_como_blancas': len(partidas_blancas),
                'partidas_como_negras': len(partidas_negras)
            }
            
            # Calcular eficiencia por color
            if len(partidas_blancas) > 0:
                metricas[agente]['eficiencia_blancas'] = (victorias_blancas + empates_blancas * 0.5) / len(partidas_blancas)
            else:
                metricas[agente]['eficiencia_blancas'] = 0
                
            if len(partidas_negras) > 0:
                metricas[agente]['eficiencia_negras'] = (victorias_negras + empates_negras * 0.5) / len(partidas_negras)
            else:
                metricas[agente]['eficiencia_negras'] = 0
        
        self.metricas_calculadas['basicas'] = metricas
        return metricas
    
    def calcular_metricas_avanzadas(self, df_resultados: pd.DataFrame) -> Dict:
        """
        Calcula métricas avanzadas de desempeño.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            
        Returns:
            dict: Métricas avanzadas por agente
        """
        logger.info("Calculando métricas avanzadas...")
        
        if 'basicas' not in self.metricas_calculadas:
            self.calcular_metricas_basicas(df_resultados)
        
        metricas_avanzadas = {}
        
        # Obtener lista de agentes
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = list(agentes_blancas.union(agentes_negras))
        
        for agente in todos_agentes:
            partidas_blancas = df_resultados[df_resultados['agente_blancas'] == agente]
            partidas_negras = df_resultados[df_resultados['agente_negras'] == agente]
            todas_partidas = pd.concat([partidas_blancas, partidas_negras])
            
            # Análisis de terminaciones
            terminaciones = todas_partidas['razon_fin'].value_counts()
            
            # Consistencia temporal
            tiempos_blancas = partidas_blancas['tiempo_promedio_blancas'].dropna()
            tiempos_negras = partidas_negras['tiempo_promedio_negras'].dropna()
            todos_tiempos = pd.concat([tiempos_blancas, tiempos_negras])
            
            consistencia_temporal = 1 / (1 + todos_tiempos.std()) if len(todos_tiempos) > 1 else 1
            
            # Análisis de duración de partidas
            movimientos = todas_partidas['movimientos_realizados']
            
            # Capacidad de finalización
            partidas_decisivas = len(todas_partidas[todas_partidas['ganador'] != 'empate'])
            tasa_finalizacion = partidas_decisivas / len(todas_partidas) if len(todas_partidas) > 0 else 0
            
            # Eficiencia versus diferentes oponentes
            oponentes_enfrentados = set()
            if len(partidas_blancas) > 0:
                oponentes_enfrentados.update(partidas_blancas['agente_negras'])
            if len(partidas_negras) > 0:
                oponentes_enfrentados.update(partidas_negras['agente_blancas'])
            
            versatilidad = len(oponentes_enfrentados)
            
            # Performance contra diferentes tipos de terminación
            victorias_jaque_mate = 0
            derrotas_jaque_mate = 0
            
            for _, partida in todas_partidas.iterrows():
                if partida['razon_fin'] == 'jaque_mate':
                    if ((partida['agente_blancas'] == agente and partida['ganador'] == 'blancas') or
                        (partida['agente_negras'] == agente and partida['ganador'] == 'negras')):
                        victorias_jaque_mate += 1
                    else:
                        derrotas_jaque_mate += 1
            
            metricas_avanzadas[agente] = {
                # Análisis de terminaciones
                'terminaciones_por_tipo': terminaciones.to_dict(),
                'victorias_por_jaque_mate': victorias_jaque_mate,
                'derrotas_por_jaque_mate': derrotas_jaque_mate,
                
                # Consistencia
                'consistencia_temporal': consistencia_temporal,
                'desviacion_tiempo': todos_tiempos.std() if len(todos_tiempos) > 1 else 0,
                
                # Eficiencia de finalización
                'tasa_finalizacion_decisiva': tasa_finalizacion,
                'movimientos_promedio_victoria': todas_partidas[
                    ((todas_partidas['agente_blancas'] == agente) & (todas_partidas['ganador'] == 'blancas')) |
                    ((todas_partidas['agente_negras'] == agente) & (todas_partidas['ganador'] == 'negras'))
                ]['movimientos_realizados'].mean(),
                
                # Versatilidad
                'oponentes_enfrentados': versatilidad,
                'adaptabilidad': versatilidad / len(todos_agentes) if len(todos_agentes) > 1 else 0,
                
                # Análisis estadístico
                'mediana_movimientos': movimientos.median(),
                'q1_movimientos': movimientos.quantile(0.25),
                'q3_movimientos': movimientos.quantile(0.75),
                'iqr_movimientos': movimientos.quantile(0.75) - movimientos.quantile(0.25)
            }
        
        self.metricas_calculadas['avanzadas'] = metricas_avanzadas
        return metricas_avanzadas
    
    def calcular_matriz_enfrentamientos(self, df_resultados: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la matriz de enfrentamientos entre agentes.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            
        Returns:
            DataFrame: Matriz de enfrentamientos
        """
        logger.info("Calculando matriz de enfrentamientos...")
        
        # Obtener lista de agentes
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = sorted(list(agentes_blancas.union(agentes_negras)))
        
        # Crear matriz
        matriz = pd.DataFrame(index=todos_agentes, columns=todos_agentes, dtype=float)
        matriz = matriz.fillna(0.0)
        
        for agente1 in todos_agentes:
            for agente2 in todos_agentes:
                if agente1 != agente2:
                    # Partidas donde agente1 es blancas y agente2 es negras
                    partidas_1_blancas = df_resultados[
                        (df_resultados['agente_blancas'] == agente1) & 
                        (df_resultados['agente_negras'] == agente2)
                    ]
                    
                    # Partidas donde agente1 es negras y agente2 es blancas
                    partidas_1_negras = df_resultados[
                        (df_resultados['agente_negras'] == agente1) & 
                        (df_resultados['agente_blancas'] == agente2)
                    ]
                    
                    # Calcular puntos de agente1 contra agente2
                    puntos = 0
                    total_partidas = 0
                    
                    # Puntos cuando agente1 juega con blancas
                    for _, partida in partidas_1_blancas.iterrows():
                        total_partidas += 1
                        if partida['ganador'] == 'blancas':
                            puntos += 1
                        elif partida['ganador'] == 'empate':
                            puntos += 0.5
                    
                    # Puntos cuando agente1 juega con negras
                    for _, partida in partidas_1_negras.iterrows():
                        total_partidas += 1
                        if partida['ganador'] == 'negras':
                            puntos += 1
                        elif partida['ganador'] == 'empate':
                            puntos += 0.5
                    
                    # Calcular porcentaje
                    if total_partidas > 0:
                        matriz.loc[agente1, agente2] = (puntos / total_partidas) * 100
                    else:
                        matriz.loc[agente1, agente2] = np.nan
        
        return matriz
    
    def calcular_rating_elo(self, df_resultados: pd.DataFrame, rating_inicial: float = 1500) -> Dict:
        """
        Calcula ratings ELO para todos los agentes.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            rating_inicial: Rating ELO inicial para todos los agentes
            
        Returns:
            dict: Ratings ELO finales por agente
        """
        logger.info("Calculando ratings ELO...")
        
        # Obtener lista de agentes
        agentes_blancas = set(df_resultados['agente_blancas'])
        agentes_negras = set(df_resultados['agente_negras'])
        todos_agentes = list(agentes_blancas.union(agentes_negras))
        
        # Inicializar ratings
        ratings = {agente: rating_inicial for agente in todos_agentes}
        
        # Procesar partidas en orden cronológico
        df_ordenado = df_resultados.sort_values(['numero_partida'], na_position='last')
        
        K = 32  # Factor K para ELO
        
        for _, partida in df_ordenado.iterrows():
            agente_blancas = partida['agente_blancas']
            agente_negras = partida['agente_negras']
            resultado = partida['resultado']  # 1: ganan blancas, 0: empate, -1: ganan negras
            
            # Ratings actuales
            rating_blancas = ratings[agente_blancas]
            rating_negras = ratings[agente_negras]
            
            # Expectativa de resultado
            expectativa_blancas = 1 / (1 + 10**((rating_negras - rating_blancas) / 400))
            expectativa_negras = 1 - expectativa_blancas
            
            # Resultado real (desde perspectiva de cada jugador)
            if resultado == 1:  # Ganan blancas
                score_blancas = 1.0
                score_negras = 0.0
            elif resultado == -1:  # Ganan negras
                score_blancas = 0.0
                score_negras = 1.0
            else:  # Empate
                score_blancas = 0.5
                score_negras = 0.5
            
            # Actualizar ratings
            nuevo_rating_blancas = rating_blancas + K * (score_blancas - expectativa_blancas)
            nuevo_rating_negras = rating_negras + K * (score_negras - expectativa_negras)
            
            ratings[agente_blancas] = nuevo_rating_blancas
            ratings[agente_negras] = nuevo_rating_negras
        
        return ratings
    
    def _calcular_elo_aproximado(self, victorias: int, empates: int, derrotas: int) -> float:
        """
        Calcula una aproximación del rating ELO basado en resultados.
        
        Args:
            victorias: Número de victorias
            empates: Número de empates
            derrotas: Número de derrotas
            
        Returns:
            float: Rating ELO aproximado
        """
        total_partidas = victorias + empates + derrotas
        if total_partidas == 0:
            return 1500
        
        puntos = victorias + (empates * 0.5)
        porcentaje = puntos / total_partidas
        
        # Fórmula aproximada para convertir porcentaje a ELO
        if porcentaje == 0:
            return 1200
        elif porcentaje == 1:
            return 1800
        else:
            # Aproximación basada en la función logística del ELO
            return 1500 + 400 * np.log10(porcentaje / (1 - porcentaje))
    
    def generar_ranking(self, df_resultados: pd.DataFrame) -> pd.DataFrame:
        """
        Genera un ranking completo de agentes.
        
        Args:
            df_resultados: DataFrame con resultados de partidas
            
        Returns:
            DataFrame: Ranking de agentes
        """
        logger.info("Generando ranking de agentes...")
        
        # Calcular todas las métricas
        metricas_basicas = self.calcular_metricas_basicas(df_resultados)
        metricas_avanzadas = self.calcular_metricas_avanzadas(df_resultados)
        ratings_elo = self.calcular_rating_elo(df_resultados)
        
        # Crear DataFrame de ranking
        ranking_data = []
        
        for agente in metricas_basicas.keys():
            basicas = metricas_basicas[agente]
            avanzadas = metricas_avanzadas[agente]
            
            ranking_data.append({
                'agente': agente,
                'puntuacion': basicas['puntos'],
                'porcentaje_puntos': basicas['porcentaje_puntos'],
                'elo_rating': ratings_elo[agente],
                'total_partidas': basicas['total_partidas'],
                'tasa_victorias': basicas['tasa_victorias'] * 100,
                'tasa_empates': basicas['tasa_empates'] * 100,
                'tiempo_promedio': basicas['tiempo_promedio_por_movimiento'],
                'consistencia_temporal': avanzadas['consistencia_temporal'],
                'tasa_finalizacion': avanzadas['tasa_finalizacion_decisiva'] * 100,
                'movimientos_promedio': basicas['movimientos_promedio_por_partida']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Ordenar por puntuación ELO
        ranking_df = ranking_df.sort_values('elo_rating', ascending=False)
        ranking_df['posicion'] = range(1, len(ranking_df) + 1)
        
        # Reordenar columnas
        columnas_ordenadas = ['posicion', 'agente', 'elo_rating', 'puntuacion', 'porcentaje_puntos',
                             'total_partidas', 'tasa_victorias', 'tasa_empates', 'tiempo_promedio',
                             'consistencia_temporal', 'tasa_finalizacion', 'movimientos_promedio']
        
        return ranking_df[columnas_ordenadas]
    
    def obtener_metricas_completas(self) -> Dict:
        """
        Obtiene todas las métricas calculadas.
        
        Returns:
            dict: Todas las métricas calculadas
        """
        return self.metricas_calculadas


if __name__ == "__main__":
    # Ejemplo de uso
    print("Probando calculador de métricas...")
    
    # Crear datos de ejemplo
    data_ejemplo = {
        'agente_blancas': ['minimax', 'mcts', 'minimax', 'dqn'],
        'agente_negras': ['mcts', 'minimax', 'dqn', 'minimax'],
        'ganador': ['blancas', 'empate', 'negras', 'blancas'],
        'resultado': [1, 0, -1, 1],
        'movimientos_realizados': [45, 67, 34, 56],
        'tiempo_promedio_blancas': [2.1, 1.8, 2.3, 1.5],
        'tiempo_promedio_negras': [3.2, 2.1, 1.9, 2.8],
        'tiempo_max_blancas': [5.1, 4.2, 6.1, 3.8],
        'tiempo_max_negras': [7.2, 5.1, 4.9, 6.2],
        'razon_fin': ['jaque_mate', 'ahogado', 'jaque_mate', 'jaque_mate'],
        'numero_partida': [1, 2, 3, 4]
    }
    
    df_ejemplo = pd.DataFrame(data_ejemplo)
    
    calculador = CalculadorMetricas()
    
    # Calcular métricas
    metricas_basicas = calculador.calcular_metricas_basicas(df_ejemplo)
    metricas_avanzadas = calculador.calcular_metricas_avanzadas(df_ejemplo)
    matriz_enfrentamientos = calculador.calcular_matriz_enfrentamientos(df_ejemplo)
    ratings_elo = calculador.calcular_rating_elo(df_ejemplo)
    ranking = calculador.generar_ranking(df_ejemplo)
    
    print("Métricas básicas:", metricas_basicas)
    print("Ratings ELO:", ratings_elo)
    print("Ranking:")
    print(ranking)
    
    print("Prueba completada")
