"""
Agente Monte Carlo Tree Search (MCTS) para ajedrez.

Este módulo implementa un agente que utiliza Monte Carlo Tree Search
para evaluar posiciones y seleccionar las mejores jugadas mediante
simulaciones aleatorias.
"""

import logging
import time
import math
import random
import numpy as np
import chess
from typing import Optional, Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodoMCTS:
    """
    Nodo en el árbol de búsqueda MCTS.
    """
    
    def __init__(self, tablero: chess.Board, movimiento: Optional[chess.Move] = None, 
                 padre: Optional['NodoMCTS'] = None):
        """
        Inicializa un nodo MCTS.
        
        Args:
            tablero: Estado del tablero en este nodo
            movimiento: Movimiento que llevó a este nodo
            padre: Nodo padre
        """
        self.tablero = tablero.copy()
        self.movimiento = movimiento
        self.padre = padre
        self.hijos: List['NodoMCTS'] = []
        self.visitas = 0
        self.victorias = 0.0
        self.movimientos_no_explorados = list(tablero.legal_moves)
        self.es_terminal = tablero.is_game_over()
    
    def es_completamente_expandido(self) -> bool:
        """Retorna True si todos los movimientos han sido explorados."""
        return len(self.movimientos_no_explorados) == 0
    
    def agregar_hijo(self, tablero: chess.Board, movimiento: chess.Move) -> 'NodoMCTS':
        """
        Agrega un nodo hijo.
        
        Args:
            tablero: Estado del tablero del hijo
            movimiento: Movimiento hacia el hijo
            
        Returns:
            Nodo hijo creado
        """
        hijo = NodoMCTS(tablero, movimiento, self)
        self.hijos.append(hijo)
        return hijo
    
    def obtener_valor_ucb1(self, c: float = 1.414) -> float:
        """
        Calcula el valor UCB1 para selección.
        
        Args:
            c: Constante de exploración
            
        Returns:
            Valor UCB1
        """
        if self.visitas == 0:
            return float('inf')
        
        explotacion = self.victorias / self.visitas
        exploracion = c * math.sqrt(math.log(self.padre.visitas) / self.visitas)
        
        return explotacion + exploracion
    
    def actualizar(self, resultado: float):
        """
        Actualiza las estadísticas del nodo.
        
        Args:
            resultado: Resultado de la simulación (0-1)
        """
        self.visitas += 1
        self.victorias += resultado


class AgenteMCTS:
    """
    Agente que utiliza Monte Carlo Tree Search para jugar ajedrez.
    
    Este agente construye un árbol de búsqueda mediante simulaciones
    aleatorias para estimar el valor de diferentes movimientos.
    """
    
    def __init__(self, num_simulaciones: int = 1000, tiempo_limite: float = 5.0, 
                 c_exploracion: float = 1.414):
        """
        Inicializa el agente MCTS.
        
        Args:
            num_simulaciones: Número de simulaciones por jugada
            tiempo_limite: Tiempo límite en segundos por jugada
            c_exploracion: Constante de exploración para UCB1
        """
        self.num_simulaciones = num_simulaciones
        self.tiempo_limite = tiempo_limite
        self.c_exploracion = c_exploracion
        self.nombre = "MCTS"
        self.estadisticas = {
            'simulaciones_realizadas': 0,
            'nodos_creados': 0,
            'tiempo_total': 0.0,
            'jugadas_realizadas': 0,
            'profundidad_maxima': 0
        }
        
        logger.info("Agente MCTS inicializado - Simulaciones: %d, Tiempo límite: %.1fs", 
                   num_simulaciones, tiempo_limite)
    
    def seleccionar_accion(self, observacion: np.ndarray, acciones_legales: np.ndarray, 
                          info: Dict[str, Any]) -> int:
        """
        Selecciona la mejor acción usando MCTS.
        
        Args:
            observacion: Estado actual del juego
            acciones_legales: Acciones legales disponibles
            info: Información adicional del entorno
            
        Returns:
            Índice de la acción seleccionada
        """
        inicio_tiempo = time.time()
        
        try:
            # Resetear estadísticas para esta jugada
            simulaciones_jugada = 0
            nodos_jugada = 0
            
            if len(acciones_legales) == 0:
                logger.warning("No hay acciones legales disponibles")
                return 0
            
            if len(acciones_legales) == 1:
                logger.info("Solo una acción legal disponible")
                return acciones_legales[0]
            
            # Obtener tablero actual
            tablero = self._obtener_tablero_desde_info(info)
            if tablero is None:
                logger.warning("No se pudo obtener estado del tablero, selección aleatoria")
                return np.random.choice(acciones_legales)
            
            # Crear nodo raíz
            raiz = NodoMCTS(tablero)
            nodos_jugada += 1
            
            # Ejecutar simulaciones MCTS
            for simulacion in range(self.num_simulaciones):
                if time.time() - inicio_tiempo > self.tiempo_limite:
                    logger.warning("Tiempo límite alcanzado en simulación %d", simulacion)
                    break
                
                # Fases de MCTS
                nodo_hoja = self._seleccionar(raiz)
                if not nodo_hoja.es_terminal:
                    nodo_hoja = self._expandir(nodo_hoja)
                    nodos_jugada += 1
                
                resultado = self._simular(nodo_hoja, inicio_tiempo)
                self._retropropagar(nodo_hoja, resultado)
                
                simulaciones_jugada += 1
            
            # Seleccionar mejor hijo
            mejor_hijo = self._seleccionar_mejor_hijo(raiz)
            mejor_accion = self._movimiento_a_accion(mejor_hijo.movimiento, tablero, acciones_legales)
            
            tiempo_jugada = time.time() - inicio_tiempo
            self.estadisticas['simulaciones_realizadas'] += simulaciones_jugada
            self.estadisticas['nodos_creados'] += nodos_jugada
            self.estadisticas['tiempo_total'] += tiempo_jugada
            self.estadisticas['jugadas_realizadas'] += 1
            
            logger.debug("MCTS - Mejor acción: %s, Simulaciones: %d, Nodos: %d, Tiempo: %.3fs",
                        mejor_accion, simulaciones_jugada, nodos_jugada, tiempo_jugada)
            
            return mejor_accion if mejor_accion is not None else acciones_legales[0]
            
        except Exception as e:
            logger.error("Error en selección de acción MCTS: %s", str(e))
            return np.random.choice(acciones_legales)
    
    def _seleccionar(self, nodo: NodoMCTS) -> NodoMCTS:
        """
        Fase de selección: navega desde la raíz hasta un nodo hoja.
        
        Args:
            nodo: Nodo desde donde comenzar la selección
            
        Returns:
            Nodo hoja seleccionado
        """
        while not nodo.es_terminal and nodo.es_completamente_expandido():
            # Seleccionar hijo con mayor valor UCB1
            mejor_hijo = max(nodo.hijos, key=lambda h: h.obtener_valor_ucb1(self.c_exploracion))
            nodo = mejor_hijo
        
        return nodo
    
    def _expandir(self, nodo: NodoMCTS) -> NodoMCTS:
        """
        Fase de expansión: agrega un nuevo nodo hijo.
        
        Args:
            nodo: Nodo a expandir
            
        Returns:
            Nuevo nodo hijo creado
        """
        if nodo.es_terminal or len(nodo.movimientos_no_explorados) == 0:
            return nodo
        
        # Seleccionar un movimiento no explorado
        movimiento = nodo.movimientos_no_explorados.pop()
        
        # Crear nuevo tablero con el movimiento
        nuevo_tablero = nodo.tablero.copy()
        nuevo_tablero.push(movimiento)
        
        # Crear y agregar hijo
        hijo = nodo.agregar_hijo(nuevo_tablero, movimiento)
        
        return hijo
    
    def _simular(self, nodo: NodoMCTS, inicio_tiempo: float) -> float:
        """
        Fase de simulación: juega aleatoriamente hasta el final.
        
        Args:
            nodo: Nodo desde donde comenzar la simulación
            inicio_tiempo: Tiempo de inicio para control de tiempo
            
        Returns:
            Resultado de la simulación (0-1)
        """
        tablero_simulacion = nodo.tablero.copy()
        jugador_inicial = tablero_simulacion.turn
        
        # Simulación con movimientos aleatorios
        max_movimientos = 200  # Límite para evitar simulaciones infinitas
        movimientos_simulados = 0
        
        while not tablero_simulacion.is_game_over() and movimientos_simulados < max_movimientos:
            if time.time() - inicio_tiempo > self.tiempo_limite:
                break
            
            movimientos_legales = list(tablero_simulacion.legal_moves)
            if not movimientos_legales:
                break
            
            # Seleccionar movimiento aleatorio con sesgo hacia capturas
            movimiento = self._seleccionar_movimiento_simulacion(tablero_simulacion, movimientos_legales)
            tablero_simulacion.push(movimiento)
            movimientos_simulados += 1
        
        # Evaluar resultado
        return self._evaluar_resultado_simulacion(tablero_simulacion, jugador_inicial)
    
    def _seleccionar_movimiento_simulacion(self, tablero: chess.Board, 
                                         movimientos: List[chess.Move]) -> chess.Move:
        """
        Selecciona un movimiento para la simulación con sesgo hacia capturas.
        
        Args:
            tablero: Estado del tablero
            movimientos: Movimientos legales disponibles
            
        Returns:
            Movimiento seleccionado
        """
        # Separar capturas de otros movimientos
        capturas = [m for m in movimientos if tablero.is_capture(m)]
        
        # Preferir capturas con probabilidad 0.7
        if capturas and random.random() < 0.7:
            return random.choice(capturas)
        else:
            return random.choice(movimientos)
    
    def _evaluar_resultado_simulacion(self, tablero: chess.Board, jugador_inicial: bool) -> float:
        """
        Evalúa el resultado de una simulación.
        
        Args:
            tablero: Estado final del tablero
            jugador_inicial: Color del jugador que inició la simulación
            
        Returns:
            Valor del resultado (0-1)
        """
        if tablero.is_checkmate():
            # Si es jaque mate, el jugador actual perdió
            ganador = not tablero.turn
            return 1.0 if ganador == jugador_inicial else 0.0
        
        if tablero.is_stalemate() or tablero.is_insufficient_material():
            return 0.5  # Empate
        
        # Si la simulación se cortó, evaluar posición
        return self._evaluar_posicion_heuristica(tablero, jugador_inicial)
    
    def _evaluar_posicion_heuristica(self, tablero: chess.Board, jugador_inicial: bool) -> float:
        """
        Evaluación heurística rápida para simulaciones cortadas.
        
        Args:
            tablero: Estado del tablero
            jugador_inicial: Color del jugador inicial
            
        Returns:
            Valor de evaluación normalizado (0-1)
        """
        valores_piezas = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        material_blancas = 0
        material_negras = 0
        
        for cuadro in chess.SQUARES:
            pieza = tablero.piece_at(cuadro)
            if pieza:
                valor = valores_piezas[pieza.piece_type]
                if pieza.color == chess.WHITE:
                    material_blancas += valor
                else:
                    material_negras += valor
        
        if material_blancas + material_negras == 0:
            return 0.5
        
        ventaja_blancas = (material_blancas - material_negras) / (material_blancas + material_negras)
        ventaja_jugador = ventaja_blancas if jugador_inicial == chess.WHITE else -ventaja_blancas
        
        # Normalizar a rango [0, 1]
        return max(0.0, min(1.0, 0.5 + ventaja_jugador * 0.5))
    
    def _retropropagar(self, nodo: NodoMCTS, resultado: float):
        """
        Fase de retropropagación: actualiza estadísticas hacia la raíz.
        
        Args:
            nodo: Nodo desde donde comenzar la retropropagación
            resultado: Resultado a propagar
        """
        while nodo is not None:
            nodo.actualizar(resultado)
            # Alternar perspectiva del resultado
            resultado = 1.0 - resultado
            nodo = nodo.padre
    
    def _seleccionar_mejor_hijo(self, raiz: NodoMCTS) -> NodoMCTS:
        """
        Selecciona el mejor hijo basado en el número de visitas.
        
        Args:
            raiz: Nodo raíz
            
        Returns:
            Mejor nodo hijo
        """
        if not raiz.hijos:
            return raiz
        
        # Seleccionar hijo con más visitas (más robusto)
        return max(raiz.hijos, key=lambda h: h.visitas)
    
    def _obtener_tablero_desde_info(self, info: Dict[str, Any]) -> Optional[chess.Board]:
        """
        Obtiene el estado del tablero desde la información del entorno.
        
        Args:
            info: Información del entorno
            
        Returns:
            Objeto Board de chess o None
        """
        try:
            if 'board' in info:
                return info['board']
            
            if 'fen' in info:
                return chess.Board(info['fen'])
            
            return chess.Board()
            
        except Exception as e:
            logger.error("Error al obtener tablero desde info: %s", str(e))
            return None
    
    def _movimiento_a_accion(self, movimiento: chess.Move, tablero: chess.Board, 
                           acciones_legales: np.ndarray) -> Optional[int]:
        """
        Convierte un movimiento a índice de acción.
        
        Args:
            movimiento: Movimiento de chess
            tablero: Estado del tablero
            acciones_legales: Acciones legales disponibles
            
        Returns:
            Índice de acción o None
        """
        try:
            movimientos_legales = list(tablero.legal_moves)
            if movimiento in movimientos_legales:
                indice = movimientos_legales.index(movimiento)
                if indice < len(acciones_legales):
                    return acciones_legales[indice]
            return None
            
        except Exception as e:
            logger.error("Error al convertir movimiento a acción: %s", str(e))
            return None
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene las estadísticas del agente.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.estadisticas.copy()
        if stats['jugadas_realizadas'] > 0:
            stats['simulaciones_promedio'] = stats['simulaciones_realizadas'] / stats['jugadas_realizadas']
            stats['tiempo_promedio_por_jugada'] = stats['tiempo_total'] / stats['jugadas_realizadas']
            stats['nodos_promedio'] = stats['nodos_creados'] / stats['jugadas_realizadas']
        else:
            stats['simulaciones_promedio'] = 0
            stats['tiempo_promedio_por_jugada'] = 0
            stats['nodos_promedio'] = 0
        
        return stats
    
    def obtener_mejor_jugada(self, tablero: chess.Board) -> chess.Move:
        """
        Obtiene la mejor jugada para un tablero de chess directamente.
        
        Args:
            tablero: Tablero de chess
            
        Returns:
            Mejor movimiento encontrado
        """
        inicio_tiempo = time.time()
        
        try:
            # Resetear estadísticas para esta jugada
            simulaciones_jugada = 0
            nodos_jugada = 0
            
            movimientos_legales = list(tablero.legal_moves)
            
            if len(movimientos_legales) == 0:
                logger.warning("No hay movimientos legales disponibles")
                return None
            
            if len(movimientos_legales) == 1:
                logger.info("Solo un movimiento legal disponible")
                return movimientos_legales[0]
            
            # Crear nodo raíz
            raiz = NodoMCTS(tablero)
            nodos_jugada += 1
            
            # Ejecutar simulaciones MCTS
            for simulacion in range(self.num_simulaciones):
                if time.time() - inicio_tiempo > self.tiempo_limite:
                    logger.warning("Tiempo límite alcanzado en simulación %d", simulacion)
                    break
                
                # Fases de MCTS
                nodo_hoja = self._seleccionar(raiz)
                if not nodo_hoja.es_terminal:
                    nodo_hoja = self._expandir(nodo_hoja)
                    nodos_jugada += 1
                
                resultado = self._simular(nodo_hoja, inicio_tiempo)
                self._retropropagar(nodo_hoja, resultado)
                
                simulaciones_jugada += 1
            
            # Seleccionar mejor hijo
            mejor_hijo = self._seleccionar_mejor_hijo(raiz)
            
            tiempo_jugada = time.time() - inicio_tiempo
            self.estadisticas['simulaciones_realizadas'] += simulaciones_jugada
            self.estadisticas['nodos_creados'] += nodos_jugada
            self.estadisticas['tiempo_total'] += tiempo_jugada
            self.estadisticas['jugadas_realizadas'] += 1
            
            logger.debug("MCTS - Mejor jugada: %s, Simulaciones: %d, Nodos: %d, Tiempo: %.3fs",
                        mejor_hijo.movimiento.uci() if mejor_hijo and mejor_hijo.movimiento else "None", 
                        simulaciones_jugada, nodos_jugada, tiempo_jugada)
            
            return mejor_hijo.movimiento if mejor_hijo and mejor_hijo.movimiento else movimientos_legales[0]
            
        except Exception as e:
            logger.error("Error en obtener_mejor_jugada MCTS: %s", str(e))
            return movimientos_legales[0] if movimientos_legales else None
    
    def reiniciar_estadisticas(self):
        """Reinicia las estadísticas del agente."""
        self.estadisticas = {
            'simulaciones_realizadas': 0,
            'nodos_creados': 0,
            'tiempo_total': 0.0,
            'jugadas_realizadas': 0,
            'profundidad_maxima': 0
        }
        logger.info("Estadísticas de MCTS reiniciadas")
    
    def __str__(self) -> str:
        return f"MCTS(simulaciones={self.num_simulaciones}, tiempo_limite={self.tiempo_limite}s)"