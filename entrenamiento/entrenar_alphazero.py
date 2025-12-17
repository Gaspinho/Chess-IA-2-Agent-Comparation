"""
Script de entrenamiento para el agente AlphaZero.

Este módulo implementa el proceso de entrenamiento de AlphaZero mediante
auto-juego y mejora iterativa de la red neuronal.
"""

import logging
import os
import time
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentes.agente_alphazero import AgenteAlphaZero, NodoMCTSAlphaZero
from entorno.entorno_ajedrez import EntornoAjedrez

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntrenadorAlphaZero:
    """
    Entrenador para el agente AlphaZero usando auto-juego.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el entrenador.
        
        Args:
            config: Configuración de entrenamiento
        """
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_simulaciones = config.get('num_simulaciones', 800)
        self.batch_size = config.get('batch_size', 64)
        self.num_iteraciones = config.get('num_iteraciones', 100)
        self.num_episodios = config.get('num_episodios', 100)
        self.temperatura = config.get('temperatura', 1.0)
        self.c_puct = config.get('c_puct', 1.5)
        self.frecuencia_guardado = config.get('frecuencia_guardado', 10)
        self.ruta_guardado = config.get('ruta_guardado', 'modelos/')
        
        # Crear directorio de modelos si no existe
        os.makedirs(self.ruta_guardado, exist_ok=True)
        
        # Inicializar agente
        self.agente = AgenteAlphaZero(
            num_simulaciones=self.num_simulaciones,
            c_puct=self.c_puct,
            temperatura=self.temperatura
        )
        
        # Optimizador
        self.optimizador = optim.Adam(
            self.agente.red_neuronal.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Buffer de experiencia
        self.buffer_experiencia = deque(maxlen=10000)
        
        # Métricas
        self.historial_perdidas = []
        self.historial_victorias = []
        
        logger.info("Entrenador AlphaZero inicializado")
        logger.info("Configuración: %s", config)
    
    def entrenar(self) -> Dict[str, Any]:
        """
        Ejecuta el proceso completo de entrenamiento.
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info("=== INICIANDO ENTRENAMIENTO ALPHAZERO ===")
        inicio_tiempo = time.time()
        
        try:
            for iteracion in range(self.num_iteraciones):
                logger.info("--- Iteración %d/%d ---", iteracion + 1, self.num_iteraciones)
                
                # Fase 1: Auto-juego
                logger.info("Generando datos de auto-juego...")
                datos_juego = self._generar_datos_autojuego()
                self.buffer_experiencia.extend(datos_juego)
                logger.info("Datos generados: %d ejemplos (Buffer total: %d)", 
                           len(datos_juego), len(self.buffer_experiencia))
                
                # Fase 2: Entrenamiento de la red
                if len(self.buffer_experiencia) >= self.batch_size:
                    logger.info("Entrenando red neuronal...")
                    perdidas = self._entrenar_red()
                    self.historial_perdidas.extend(perdidas)
                    logger.info("Pérdida promedio: %.4f", np.mean(perdidas))
                
                # Guardar modelo periódicamente
                if (iteracion + 1) % self.frecuencia_guardado == 0:
                    ruta_modelo = os.path.join(
                        self.ruta_guardado, 
                        f'alphazero_iter_{iteracion + 1}.pth'
                    )
                    self.agente.guardar_modelo(ruta_modelo)
                    logger.info("Modelo guardado: %s", ruta_modelo)
                
                # Mostrar estadísticas
                self._mostrar_estadisticas(iteracion + 1)
            
            # Guardar modelo final
            ruta_final = os.path.join(self.ruta_guardado, 'alphazero_final.pth')
            self.agente.guardar_modelo(ruta_final)
            
            tiempo_total = time.time() - inicio_tiempo
            logger.info("=== ENTRENAMIENTO COMPLETADO ===")
            logger.info("Tiempo total: %.2f minutos", tiempo_total / 60)
            
            return {
                'entrenamiento_exitoso': True,
                'iteraciones_completadas': self.num_iteraciones,
                'tiempo_total': tiempo_total,
                'perdida_final': np.mean(self.historial_perdidas[-10:]) if self.historial_perdidas else 0,
                'ruta_modelo': ruta_final
            }
            
        except Exception as e:
            logger.error("Error durante el entrenamiento: %s", str(e), exc_info=True)
            return {
                'entrenamiento_exitoso': False,
                'error': str(e)
            }
    
    def _generar_datos_autojuego(self) -> List[Tuple]:
        """
        Genera datos mediante auto-juego.
        
        Returns:
            Lista de tuplas (estado, política, resultado)
        """
        datos = []
        
        logger.info("Iniciando %d episodios de auto-juego...", self.num_episodios)
        inicio_generacion = time.time()
        
        for episodio in range(self.num_episodios):
            # Mostrar progreso cada 10 episodios
            if episodio > 0 and episodio % 10 == 0:
                tiempo_transcurrido = time.time() - inicio_generacion
                tiempo_por_episodio = tiempo_transcurrido / episodio
                episodios_restantes = self.num_episodios - episodio
                tiempo_estimado = tiempo_por_episodio * episodios_restantes
                logger.info("  Progreso: %d/%d episodios (%.1f%%) - Tiempo estimado restante: %.1f min", 
                           episodio, self.num_episodios, 
                           (episodio/self.num_episodios)*100,
                           tiempo_estimado / 60)
            
            # Jugar una partida completa
            try:
                estados_partida, politicas_partida, tablero_final = self._jugar_partida()
            except Exception as e:
                logger.error("Error en episodio %d: %s", episodio, str(e))
                continue
            
            # Obtener resultado de la partida desde perspectiva de blancas
            turno_inicial = chess.WHITE
            resultado = self._evaluar_resultado(estados_partida[-1], tablero_final, turno_inicial)
            
            # Agregar datos con el resultado (alternando perspectiva por turno)
            for idx, (estado, politica) in enumerate(zip(estados_partida, politicas_partida)):
                # Alternar signo según el turno (jugada par=blancas, impar=negras)
                resultado_turno = resultado if idx % 2 == 0 else -resultado
                datos.append((estado, politica, resultado_turno))
            
            if (episodio + 1) % 10 == 0:
                logger.debug("Episodios completados: %d/%d", episodio + 1, self.num_episodios)
        
        return datos
    
    def _jugar_partida(self) -> Tuple[List, List, chess.Board]:
        """
        Juega una partida completa de auto-juego.
        
        Returns:
            Tupla (estados, políticas, tablero_final)
        """
        tablero = chess.Board()
        estados = []
        politicas = []
        
        max_jugadas = 200  # Límite para evitar partidas infinitas
        jugadas = 0
        
        logger.debug("  Iniciando nueva partida (max %d jugadas, %d simulaciones por jugada)...", 
                    max_jugadas, self.num_simulaciones)
        
        while not tablero.is_game_over() and jugadas < max_jugadas:
            # Ejecutar MCTS desde este estado
            raiz = NodoMCTSAlphaZero(tablero)
            
            # Mostrar progreso cada 20 jugadas
            if jugadas > 0 and jugadas % 20 == 0:
                logger.debug("    Jugada %d/%d en partida actual", jugadas, max_jugadas)
            
            for sim in range(self.num_simulaciones):
                self._ejecutar_simulacion_mcts(raiz)
                # Mostrar progreso cada 200 simulaciones si está en modo verbose
                if sim > 0 and sim % 200 == 0:
                    logger.debug("      Simulaciones: %d/%d", sim, self.num_simulaciones)
            
            # Guardar estado y política
            estado = self._tablero_a_array(tablero)
            politica = self._extraer_politica(raiz, tablero)
            
            estados.append(estado)
            politicas.append(politica)
            
            # Seleccionar movimiento
            movimiento = self._seleccionar_movimiento_autojuego(raiz)
            if movimiento is None:
                break
            
            tablero.push(movimiento)
            jugadas += 1
        
        logger.debug("  Partida finalizada: %d jugadas, resultado: %s", 
                    jugadas, tablero.result() if tablero.is_game_over() else "Límite alcanzado")
        
        return estados, politicas, tablero
    
    def _ejecutar_simulacion_mcts(self, raiz: NodoMCTSAlphaZero):
        """
        Ejecuta una simulación MCTS.
        
        Args:
            raiz: Nodo raíz
        """
        nodo = raiz
        tablero_busqueda = raiz.tablero.copy()
        
        # Selección
        while not nodo.es_hoja() and not nodo.es_terminal:
            movimiento, nodo = self._seleccionar_hijo_mcts(nodo)
            tablero_busqueda.push(movimiento)
        
        # Expansión y evaluación
        valor = self._expandir_y_evaluar_nodo(nodo, tablero_busqueda)
        
        # Retropropagación
        while nodo is not None:
            nodo.actualizar(valor)
            valor = -valor
            nodo = nodo.padre
    
    def _seleccionar_hijo_mcts(self, nodo: NodoMCTSAlphaZero) -> Tuple:
        """Selecciona el mejor hijo según UCB."""
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
    
    def _expandir_y_evaluar_nodo(self, nodo: NodoMCTSAlphaZero, 
                                  tablero: chess.Board) -> float:
        """Expande y evalúa un nodo con la red neuronal."""
        if nodo.es_terminal:
            resultado = tablero.result()
            if resultado == "1-0":
                return 1.0 if tablero.turn == chess.WHITE else -1.0
            elif resultado == "0-1":
                return -1.0 if tablero.turn == chess.WHITE else 1.0
            else:
                return 0.0
        
        # Evaluar con la red
        estado = self.agente._tablero_a_tensor(tablero)
        
        with torch.no_grad():
            politica_logits, valor = self.agente.red_neuronal(estado)
        
        politica = torch.exp(politica_logits).cpu().numpy()[0]
        valor = valor.item()
        
        # Expandir
        probabilidades = self.agente._mapear_politica_a_movimientos(tablero, politica)
        nodo.expandir(probabilidades)
        
        return valor
    
    def _seleccionar_movimiento_autojuego(self, raiz: NodoMCTSAlphaZero) -> chess.Move:
        """Selecciona movimiento durante auto-juego."""
        if len(raiz.hijos) == 0:
            return None
        
        movimientos = list(raiz.hijos.keys())
        visitas = np.array([raiz.hijos[m].visitas for m in movimientos])
        
        # Aplicar temperatura
        if self.temperatura == 0:
            return movimientos[np.argmax(visitas)]
        else:
            visitas = visitas ** (1.0 / self.temperatura)
            probabilidades = visitas / visitas.sum()
            return np.random.choice(movimientos, p=probabilidades)
    
    def _tablero_a_array(self, tablero: chess.Board) -> np.ndarray:
        """Convierte tablero a array numpy."""
        return self.agente._tablero_a_tensor(tablero).cpu().numpy()[0]
    
    def _extraer_politica(self, raiz: NodoMCTSAlphaZero, 
                         tablero: chess.Board) -> np.ndarray:
        """Extrae distribución de política desde las visitas MCTS."""
        politica = np.zeros(4672, dtype=np.float32)
        
        if len(raiz.hijos) == 0:
            return politica
        
        visitas_totales = sum(hijo.visitas for hijo in raiz.hijos.values())
        
        if visitas_totales > 0:
            for movimiento, hijo in raiz.hijos.items():
                idx = movimiento.from_square * 64 + movimiento.to_square
                idx = min(idx, 4671)
                politica[idx] = hijo.visitas / visitas_totales
        
        return politica
    
    def _evaluar_resultado(self, estado_final: np.ndarray, tablero_final: chess.Board, turno_inicial: chess.Color) -> float:
        """Evalúa el resultado de una partida desde la perspectiva del turno inicial.
        
        Args:
            estado_final: Estado final (no usado, para compatibilidad)
            tablero_final: Tablero al final de la partida
            turno_inicial: Color que inició la partida
            
        Returns:
            Resultado: 1.0 si ganó turno_inicial, -1.0 si perdió, 0.0 empate
        """
        if not tablero_final.is_game_over():
            # Si no terminó, considerar empate
            return 0.0
        
        resultado = tablero_final.result()
        
        if resultado == "1-0":  # Blancas ganan
            return 1.0 if turno_inicial == chess.WHITE else -1.0
        elif resultado == "0-1":  # Negras ganan
            return 1.0 if turno_inicial == chess.BLACK else -1.0
        else:  # Empate ("1/2-1/2" o "*")
            return 0.0
    
    def _entrenar_red(self) -> List[float]:
        """
        Entrena la red neuronal con datos del buffer.
        
        Returns:
            Lista de pérdidas
        """
        self.agente.red_neuronal.train()
        perdidas = []
        
        num_batches = min(50, len(self.buffer_experiencia) // self.batch_size)
        
        for _ in range(num_batches):
            # Muestrear batch
            batch = random.sample(self.buffer_experiencia, self.batch_size)
            estados, politicas_objetivo, valores_objetivo = zip(*batch)
            
            # Convertir a tensors
            estados_tensor = torch.FloatTensor(np.array(estados)).to(self.agente.dispositivo)
            politicas_tensor = torch.FloatTensor(np.array(politicas_objetivo)).to(self.agente.dispositivo)
            valores_tensor = torch.FloatTensor(np.array(valores_objetivo)).unsqueeze(1).to(self.agente.dispositivo)
            
            # Forward pass
            politica_pred, valor_pred = self.agente.red_neuronal(estados_tensor)
            
            # Calcular pérdidas
            perdida_valor = nn.MSELoss()(valor_pred, valores_tensor)
            perdida_politica = -torch.mean(torch.sum(politicas_tensor * politica_pred, dim=1))
            perdida_total = perdida_valor + perdida_politica
            
            # Backward pass
            self.optimizador.zero_grad()
            perdida_total.backward()
            torch.nn.utils.clip_grad_norm_(self.agente.red_neuronal.parameters(), 1.0)
            self.optimizador.step()
            
            perdidas.append(perdida_total.item())
        
        self.agente.red_neuronal.eval()
        return perdidas
    
    def _mostrar_estadisticas(self, iteracion: int):
        """Muestra estadísticas de entrenamiento."""
        if self.historial_perdidas:
            perdida_reciente = np.mean(self.historial_perdidas[-50:])
            logger.info("Iteración %d - Pérdida reciente: %.4f", iteracion, perdida_reciente)


def entrenar_agente_alphazero(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función principal para entrenar el agente AlphaZero.
    
    Args:
        config: Configuración de entrenamiento
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    logger.info("Iniciando entrenamiento de AlphaZero")
    
    try:
        entrenador = EntrenadorAlphaZero(config)
        resultados = entrenador.entrenar()
        return resultados
    except Exception as e:
        logger.error("Error al entrenar AlphaZero: %s", str(e))
        return {
            'entrenamiento_exitoso': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Configuración de prueba
    config_prueba = {
        'learning_rate': 0.001,
        'num_simulaciones': 100,
        'batch_size': 32,
        'num_iteraciones': 5,
        'num_episodios': 10,
        'temperatura': 1.0,
        'c_puct': 1.5,
        'frecuencia_guardado': 2,
        'ruta_guardado': 'modelos/'
    }
    
    resultado = entrenar_agente_alphazero(config_prueba)
    print("\nResultado del entrenamiento:")
    print(resultado)
