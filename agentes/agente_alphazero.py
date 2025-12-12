"""
Agente AlphaZero para ajedrez.

Este módulo implementa una versión simplificada del algoritmo AlphaZero
que combina Monte Carlo Tree Search con redes neuronales profundas.
"""

import logging
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from typing import Optional, Dict, Any, List, Tuple
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedNeuralAlphaZero(nn.Module):
    """
    Red neuronal para AlphaZero.
    
    Arquitectura:
    - Torre de capas convolucionales residuales
    - Dos cabezas: política (movimientos) y valor (evaluación de posición)
    """
    
    def __init__(self, num_canales: int = 256, num_bloques_res: int = 10):
        """
        Inicializa la red neuronal.
        
        Args:
            num_canales: Número de canales en capas convolucionales
            num_bloques_res: Número de bloques residuales
        """
        super(RedNeuralAlphaZero, self).__init__()
        
        # Capa de entrada (8x8x119 -> representación del tablero)
        self.conv_entrada = nn.Conv2d(119, num_canales, kernel_size=3, padding=1)
        self.bn_entrada = nn.BatchNorm2d(num_canales)
        
        # Bloques residuales
        self.bloques_res = nn.ModuleList([
            BloqueResidual(num_canales) for _ in range(num_bloques_res)
        ])
        
        # Cabeza de política (distribución de probabilidad sobre movimientos)
        self.conv_politica = nn.Conv2d(num_canales, 32, kernel_size=1)
        self.bn_politica = nn.BatchNorm2d(32)
        self.fc_politica = nn.Linear(32 * 8 * 8, 4672)  # 4672 posibles movimientos
        
        # Cabeza de valor (evaluación de posición)
        self.conv_valor = nn.Conv2d(num_canales, 32, kernel_size=1)
        self.bn_valor = nn.BatchNorm2d(32)
        self.fc_valor1 = nn.Linear(32 * 8 * 8, 256)
        self.fc_valor2 = nn.Linear(256, 1)
        
        logger.info("Red AlphaZero inicializada - Canales: %d, Bloques: %d", 
                   num_canales, num_bloques_res)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de la red.
        
        Args:
            x: Estado del tablero [batch, 119, 8, 8]
            
        Returns:
            Tupla (política, valor)
        """
        # Torre de bloques residuales
        x = F.relu(self.bn_entrada(self.conv_entrada(x)))
        for bloque in self.bloques_res:
            x = bloque(x)
        
        # Cabeza de política
        politica = F.relu(self.bn_politica(self.conv_politica(x)))
        politica = politica.view(-1, 32 * 8 * 8)
        politica = self.fc_politica(politica)
        politica = F.log_softmax(politica, dim=1)
        
        # Cabeza de valor
        valor = F.relu(self.bn_valor(self.conv_valor(x)))
        valor = valor.view(-1, 32 * 8 * 8)
        valor = F.relu(self.fc_valor1(valor))
        valor = torch.tanh(self.fc_valor2(valor))
        
        return politica, valor


class BloqueResidual(nn.Module):
    """Bloque residual para la red AlphaZero."""
    
    def __init__(self, num_canales: int):
        """
        Inicializa el bloque residual.
        
        Args:
            num_canales: Número de canales
        """
        super(BloqueResidual, self).__init__()
        self.conv1 = nn.Conv2d(num_canales, num_canales, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_canales)
        self.conv2 = nn.Conv2d(num_canales, num_canales, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_canales)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con conexión residual."""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class NodoMCTSAlphaZero:
    """
    Nodo en el árbol de búsqueda MCTS para AlphaZero.
    """
    
    def __init__(self, tablero: chess.Board, prob_previa: float = 1.0, 
                 padre: Optional['NodoMCTSAlphaZero'] = None):
        """
        Inicializa un nodo MCTS.
        
        Args:
            tablero: Estado del tablero
            prob_previa: Probabilidad a priori de la red neuronal
            padre: Nodo padre
        """
        self.tablero = tablero.copy()
        self.padre = padre
        self.hijos: Dict[chess.Move, 'NodoMCTSAlphaZero'] = {}
        self.visitas = 0
        self.valor_total = 0.0
        self.prob_previa = prob_previa
        self.es_terminal = tablero.is_game_over()
    
    def obtener_valor_ucb(self, c_puct: float = 1.5) -> float:
        """
        Calcula el valor UCB adaptado para AlphaZero.
        
        Args:
            c_puct: Constante de exploración
            
        Returns:
            Valor UCB
        """
        if self.visitas == 0:
            q_valor = 0
        else:
            q_valor = self.valor_total / self.visitas
        
        u_valor = c_puct * self.prob_previa * math.sqrt(self.padre.visitas) / (1 + self.visitas)
        
        return q_valor + u_valor
    
    def expandir(self, probabilidades_politica: Dict[chess.Move, float]):
        """
        Expande el nodo con movimientos legales.
        
        Args:
            probabilidades_politica: Probabilidades de la política para cada movimiento
        """
        for movimiento in self.tablero.legal_moves:
            if movimiento not in self.hijos:
                tablero_hijo = self.tablero.copy()
                tablero_hijo.push(movimiento)
                prob = probabilidades_politica.get(movimiento, 1e-8)
                self.hijos[movimiento] = NodoMCTSAlphaZero(tablero_hijo, prob, self)
    
    def actualizar(self, valor: float):
        """
        Actualiza las estadísticas del nodo.
        
        Args:
            valor: Valor de la evaluación
        """
        self.visitas += 1
        self.valor_total += valor
    
    def es_hoja(self) -> bool:
        """Retorna True si el nodo es una hoja (no expandido)."""
        return len(self.hijos) == 0


class AgenteAlphaZero:
    """
    Agente que utiliza AlphaZero para jugar ajedrez.
    
    Este agente combina MCTS con una red neuronal profunda que aprende
    mediante auto-juego y mejora iterativa.
    """
    
    def __init__(self, 
                 num_simulaciones: int = 800,
                 c_puct: float = 1.5,
                 temperatura: float = 1.0,
                 modelo_path: Optional[str] = None,
                 dispositivo: str = 'cpu'):
        """
        Inicializa el agente AlphaZero.
        
        Args:
            num_simulaciones: Número de simulaciones MCTS por jugada
            c_puct: Constante de exploración para UCB
            temperatura: Temperatura para muestreo de política (1.0 = exploración, 0 = explotación)
            modelo_path: Ruta al modelo pre-entrenado
            dispositivo: 'cpu' o 'cuda'
        """
        self.num_simulaciones = num_simulaciones
        self.c_puct = c_puct
        self.temperatura = temperatura
        self.nombre = "AlphaZero"
        
        # Configurar dispositivo
        self.dispositivo = torch.device(dispositivo if torch.cuda.is_available() else 'cpu')
        
        # Inicializar red neuronal
        self.red_neuronal = RedNeuralAlphaZero().to(self.dispositivo)
        
        # Cargar modelo si existe
        if modelo_path and self._cargar_modelo(modelo_path):
            logger.info("Modelo cargado desde: %s", modelo_path)
        else:
            logger.info("Usando red neuronal inicializada aleatoriamente")
        
        self.red_neuronal.eval()
        
        # Estadísticas
        self.estadisticas = {
            'simulaciones_realizadas': 0,
            'nodos_creados': 0,
            'tiempo_total': 0.0,
            'jugadas_realizadas': 0,
            'predicciones_red': 0
        }
        
        logger.info("Agente AlphaZero inicializado - Simulaciones: %d, Dispositivo: %s", 
                   num_simulaciones, self.dispositivo)
    
    def seleccionar_accion(self, observacion: np.ndarray, acciones_legales: np.ndarray, 
                          info: Dict[str, Any]) -> int:
        """
        Selecciona la mejor acción usando AlphaZero (MCTS + Red Neuronal).
        
        Args:
            observacion: Estado actual del juego
            acciones_legales: Acciones legales disponibles
            info: Información adicional del entorno
            
        Returns:
            Índice de la acción seleccionada
        """
        inicio_tiempo = time.time()
        
        try:
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
            
            # Ejecutar MCTS
            raiz = NodoMCTSAlphaZero(tablero)
            
            for _ in range(self.num_simulaciones):
                nodo = raiz
                tablero_busqueda = tablero.copy()
                
                # Selección: bajar por el árbol hasta encontrar una hoja
                while not nodo.es_hoja() and not nodo.es_terminal:
                    movimiento, nodo = self._seleccionar_hijo(nodo)
                    tablero_busqueda.push(movimiento)
                
                # Expansión y evaluación
                valor = self._expandir_y_evaluar(nodo, tablero_busqueda)
                
                # Retropropagación
                self._retropropagar(nodo, valor)
                
                self.estadisticas['simulaciones_realizadas'] += 1
            
            # Seleccionar mejor movimiento según visitas
            mejor_movimiento = self._seleccionar_mejor_movimiento(raiz)
            
            # Convertir movimiento a acción
            mejor_accion = self._movimiento_a_accion(mejor_movimiento, tablero, acciones_legales)
            
            tiempo_jugada = time.time() - inicio_tiempo
            self.estadisticas['tiempo_total'] += tiempo_jugada
            self.estadisticas['jugadas_realizadas'] += 1
            
            logger.info("AlphaZero jugada: %s (%.2fs, %d simulaciones)", 
                       mejor_movimiento.uci() if mejor_movimiento else "None",
                       tiempo_jugada, self.num_simulaciones)
            
            return mejor_accion if mejor_accion is not None else acciones_legales[0]
            
        except Exception as e:
            logger.error("Error en selección AlphaZero: %s", str(e))
            return np.random.choice(acciones_legales)
    
    def _seleccionar_hijo(self, nodo: NodoMCTSAlphaZero) -> Tuple[chess.Move, NodoMCTSAlphaZero]:
        """
        Selecciona el mejor hijo según UCB.
        
        Args:
            nodo: Nodo actual
            
        Returns:
            Tupla (movimiento, nodo_hijo)
        """
        mejor_valor = float('-inf')
        mejor_movimiento = None
        mejor_hijo = None
        
        for movimiento, hijo in nodo.hijos.items():
            valor_ucb = hijo.obtener_valor_ucb(self.c_puct)
            if valor_ucb > mejor_valor:
                mejor_valor = valor_ucb
                mejor_movimiento = movimiento
                mejor_hijo = hijo
        
        return mejor_movimiento, mejor_hijo
    
    def _expandir_y_evaluar(self, nodo: NodoMCTSAlphaZero, 
                           tablero: chess.Board) -> float:
        """
        Expande el nodo y evalúa la posición con la red neuronal.
        
        Args:
            nodo: Nodo a expandir
            tablero: Estado del tablero
            
        Returns:
            Valor de la evaluación
        """
        if nodo.es_terminal:
            # Posición terminal
            resultado = tablero.result()
            if resultado == "1-0":
                return 1.0 if tablero.turn == chess.WHITE else -1.0
            elif resultado == "0-1":
                return -1.0 if tablero.turn == chess.WHITE else 1.0
            else:
                return 0.0
        
        # Evaluar con la red neuronal
        estado = self._tablero_a_tensor(tablero)
        
        with torch.no_grad():
            politica_logits, valor = self.red_neuronal(estado)
            self.estadisticas['predicciones_red'] += 1
        
        # Convertir política a probabilidades
        politica = torch.exp(politica_logits).cpu().numpy()[0]
        valor = valor.item()
        
        # Mapear política a movimientos legales
        probabilidades_movimientos = self._mapear_politica_a_movimientos(
            tablero, politica
        )
        
        # Expandir nodo
        nodo.expandir(probabilidades_movimientos)
        self.estadisticas['nodos_creados'] += len(nodo.hijos)
        
        return valor
    
    def _retropropagar(self, nodo: NodoMCTSAlphaZero, valor: float):
        """
        Retropropaga el valor por el árbol.
        
        Args:
            nodo: Nodo desde donde retropropagar
            valor: Valor a retropropagar
        """
        while nodo is not None:
            nodo.actualizar(valor)
            valor = -valor  # Cambiar perspectiva
            nodo = nodo.padre
    
    def _seleccionar_mejor_movimiento(self, raiz: NodoMCTSAlphaZero) -> Optional[chess.Move]:
        """
        Selecciona el mejor movimiento según las visitas.
        
        Args:
            raiz: Nodo raíz
            
        Returns:
            Mejor movimiento
        """
        if len(raiz.hijos) == 0:
            return None
        
        if self.temperatura == 0:
            # Explotación pura: elegir el más visitado
            return max(raiz.hijos.items(), key=lambda x: x[1].visitas)[0]
        else:
            # Muestreo según temperatura
            movimientos = list(raiz.hijos.keys())
            visitas = np.array([raiz.hijos[m].visitas for m in movimientos])
            
            # Aplicar temperatura
            visitas = visitas ** (1.0 / self.temperatura)
            probabilidades = visitas / visitas.sum()
            
            return np.random.choice(movimientos, p=probabilidades)
    
    def _tablero_a_tensor(self, tablero: chess.Board) -> torch.Tensor:
        """
        Convierte un tablero de ajedrez a tensor para la red neuronal.
        
        Args:
            tablero: Tablero de ajedrez
            
        Returns:
            Tensor [1, 119, 8, 8]
        """
        # Representación simplificada: 119 planos
        # - 12 planos por pieza (6 tipos x 2 colores) x 8 histórico
        # - 7 planos adicionales (turno, castling, en passant, etc.)
        
        estado = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Planos de piezas (simplificado - solo posición actual)
        tipos_piezas = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN, chess.KING]
        
        for idx, tipo_pieza in enumerate(tipos_piezas):
            # Piezas blancas
            for cuadrado in tablero.pieces(tipo_pieza, chess.WHITE):
                fila, col = divmod(cuadrado, 8)
                estado[idx, fila, col] = 1.0
            
            # Piezas negras
            for cuadrado in tablero.pieces(tipo_pieza, chess.BLACK):
                fila, col = divmod(cuadrado, 8)
                estado[6 + idx, fila, col] = 1.0
        
        # Planos adicionales
        if tablero.turn == chess.WHITE:
            estado[12, :, :] = 1.0  # Turno blancas
        else:
            estado[13, :, :] = 1.0  # Turno negras
        
        # Castling rights
        if tablero.has_kingside_castling_rights(chess.WHITE):
            estado[14, :, :] = 1.0
        if tablero.has_queenside_castling_rights(chess.WHITE):
            estado[15, :, :] = 1.0
        if tablero.has_kingside_castling_rights(chess.BLACK):
            estado[16, :, :] = 1.0
        if tablero.has_queenside_castling_rights(chess.BLACK):
            estado[17, :, :] = 1.0
        
        # Convertir a tensor
        tensor = torch.from_numpy(estado).unsqueeze(0).to(self.dispositivo)
        return tensor
    
    def _mapear_politica_a_movimientos(self, tablero: chess.Board, 
                                      politica: np.ndarray) -> Dict[chess.Move, float]:
        """
        Mapea las salidas de política de la red a movimientos legales.
        
        Args:
            tablero: Tablero actual
            politica: Vector de política de la red [4672]
            
        Returns:
            Diccionario {movimiento: probabilidad}
        """
        probabilidades = {}
        movimientos_legales = list(tablero.legal_moves)
        
        if len(movimientos_legales) == 0:
            return probabilidades
        
        # Mapeo simplificado: asignar probabilidades uniformes
        # En una implementación completa, se usaría el mapeo exacto de AlphaZero
        prob_total = 0.0
        for movimiento in movimientos_legales:
            # Índice simplificado basado en from_square y to_square
            idx = movimiento.from_square * 64 + movimiento.to_square
            idx = min(idx, len(politica) - 1)
            prob = max(politica[idx], 1e-8)
            probabilidades[movimiento] = prob
            prob_total += prob
        
        # Normalizar
        if prob_total > 0:
            for movimiento in probabilidades:
                probabilidades[movimiento] /= prob_total
        
        return probabilidades
    
    def _obtener_tablero_desde_info(self, info: Dict[str, Any]) -> Optional[chess.Board]:
        """
        Obtiene el tablero desde la información del entorno.
        
        Args:
            info: Información del entorno
            
        Returns:
            Tablero o None si no se puede obtener
        """
        try:
            if 'legal_moves' in info and hasattr(info['legal_moves'], 'board'):
                return info['legal_moves'].board.copy()
            return None
        except Exception as e:
            logger.debug("No se pudo obtener tablero desde info: %s", str(e))
            return None
    
    def _movimiento_a_accion(self, movimiento: Optional[chess.Move], 
                            tablero: chess.Board, 
                            acciones_legales: np.ndarray) -> Optional[int]:
        """
        Convierte un movimiento de chess a índice de acción.
        
        Args:
            movimiento: Movimiento a convertir
            tablero: Tablero actual
            acciones_legales: Array de acciones legales
            
        Returns:
            Índice de acción o None
        """
        if movimiento is None:
            return None
        
        try:
            movimientos = list(tablero.legal_moves)
            if movimiento in movimientos:
                idx = movimientos.index(movimiento)
                if idx < len(acciones_legales):
                    return acciones_legales[idx]
        except Exception as e:
            logger.debug("Error al convertir movimiento a acción: %s", str(e))
        
        return None
    
    def _cargar_modelo(self, path: str) -> bool:
        """
        Carga un modelo pre-entrenado.
        
        Args:
            path: Ruta al archivo del modelo
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            self.red_neuronal.load_state_dict(torch.load(path, map_location=self.dispositivo))
            return True
        except Exception as e:
            logger.warning("No se pudo cargar modelo desde %s: %s", path, str(e))
            return False
    
    def guardar_modelo(self, path: str):
        """
        Guarda el modelo actual.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        try:
            torch.save(self.red_neuronal.state_dict(), path)
            logger.info("Modelo guardado en: %s", path)
        except Exception as e:
            logger.error("Error al guardar modelo: %s", str(e))
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retorna las estadísticas del agente."""
        stats = self.estadisticas.copy()
        if stats['jugadas_realizadas'] > 0:
            stats['tiempo_promedio_jugada'] = stats['tiempo_total'] / stats['jugadas_realizadas']
            stats['simulaciones_promedio'] = stats['simulaciones_realizadas'] / stats['jugadas_realizadas']
        return stats
