"""
Agente Stockfish para ajedrez.

Este módulo implementa un wrapper alrededor del motor de ajedrez Stockfish,
permitiendo que compita contra otros agentes del proyecto.
"""

import logging
import time
import os
import chess
import chess.engine
import numpy as np
from typing import Optional, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenteStockfish:
    """
    Agente que utiliza el motor Stockfish para jugar ajedrez.
    
    Stockfish es uno de los motores de ajedrez más fuertes del mundo,
    utilizando búsqueda alfa-beta altamente optimizada y evaluación avanzada.
    """
    
    def __init__(self, 
                 ruta_ejecutable: Optional[str] = None,
                 nivel_habilidad: int = 20,
                 profundidad: Optional[int] = None,
                 tiempo_limite: float = 1.0,
                 threads: int = 1,
                 hash_size: int = 128):
        """
        Inicializa el agente Stockfish.
        
        Args:
            ruta_ejecutable: Ruta al ejecutable de Stockfish. Si es None, busca en PATH
            nivel_habilidad: Nivel de habilidad (0-20, donde 20 es máximo)
            profundidad: Profundidad de búsqueda (None = sin límite, usa tiempo)
            tiempo_limite: Tiempo límite en segundos para cada jugada
            threads: Número de hilos a usar
            hash_size: Tamaño de la tabla hash en MB
        """
        self.nombre = f"Stockfish_L{nivel_habilidad}"
        self.nivel_habilidad = nivel_habilidad
        self.profundidad = profundidad
        self.tiempo_limite = tiempo_limite
        self.threads = threads
        self.hash_size = hash_size
        
        self.estadisticas = {
            'jugadas_realizadas': 0,
            'tiempo_total': 0.0,
            'evaluaciones': [],
            'profundidades_alcanzadas': []
        }
        
        # Intentar encontrar Stockfish
        self.ruta_ejecutable = self._encontrar_stockfish(ruta_ejecutable)
        self.engine = None
        
        # Inicializar motor
        self._inicializar_motor()
        
        logger.info("Agente Stockfish inicializado - Nivel: %d, Tiempo límite: %.1fs", 
                   nivel_habilidad, tiempo_limite)
    
    def _encontrar_stockfish(self, ruta: Optional[str]) -> str:
        """
        Encuentra el ejecutable de Stockfish.
        
        Args:
            ruta: Ruta proporcionada por el usuario
            
        Returns:
            Ruta válida al ejecutable
        """
        if ruta and os.path.exists(ruta):
            return ruta
        
        # Rutas comunes donde se puede encontrar Stockfish
        rutas_posibles = [
            "stockfish",  # En PATH
            "stockfish.exe",  # Windows en PATH
            "/usr/games/stockfish",  # Linux
            "/usr/local/bin/stockfish",  # macOS/Linux
            "C:\\Program Files\\Stockfish\\stockfish.exe",  # Windows común
            "C:\\Stockfish\\stockfish.exe",  # Windows alternativa
            "./stockfish",  # Directorio actual
            "./stockfish.exe",  # Directorio actual Windows
        ]
        
        for ruta_posible in rutas_posibles:
            if os.path.exists(ruta_posible):
                logger.info("Stockfish encontrado en: %s", ruta_posible)
                return ruta_posible
        
        # Si no se encuentra, intentar con el comando por defecto
        logger.warning("No se encontró Stockfish en rutas comunes. Intentando con comando 'stockfish'")
        return "stockfish"
    
    def _inicializar_motor(self):
        """Inicializa el motor de ajedrez."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.ruta_ejecutable)
            
            # Configurar opciones del motor
            self.engine.configure({
                "Skill Level": self.nivel_habilidad,
                "Threads": self.threads,
                "Hash": self.hash_size,
            })
            
            logger.info("Motor Stockfish inicializado correctamente")
            
        except FileNotFoundError:
            logger.error("No se pudo encontrar el ejecutable de Stockfish en: %s", self.ruta_ejecutable)
            logger.error("Por favor, instala Stockfish:")
            logger.error("  - Windows: Descarga desde https://stockfishchess.org/download/")
            logger.error("  - Linux: sudo apt-get install stockfish")
            logger.error("  - macOS: brew install stockfish")
            raise
        except Exception as e:
            logger.error("Error al inicializar Stockfish: %s", str(e))
            raise
    
    def seleccionar_accion(self, observacion: np.ndarray, acciones_legales: np.ndarray, 
                          info: Dict[str, Any]) -> int:
        """
        Selecciona la mejor acción usando Stockfish.
        
        Args:
            observacion: Estado actual del juego
            acciones_legales: Acciones legales disponibles
            info: Información adicional del entorno (debe contener 'tablero')
            
        Returns:
            Índice de la acción seleccionada
        """
        inicio_tiempo = time.time()
        
        try:
            if self.engine is None:
                logger.error("Motor no inicializado")
                return np.random.choice(len(acciones_legales))
            
            # Obtener tablero desde info
            tablero = info.get('tablero')
            if tablero is None:
                logger.error("No se encontró el tablero en la información")
                return np.random.choice(len(acciones_legales))
            
            # Configurar límites de búsqueda
            limite = chess.engine.Limit()
            if self.profundidad is not None:
                limite.depth = self.profundidad
            else:
                limite.time = self.tiempo_limite
            
            # Obtener mejor movimiento de Stockfish
            resultado = self.engine.play(tablero, limite)
            mejor_movimiento = resultado.move
            
            # Obtener información adicional si está disponible
            if hasattr(resultado, 'info'):
                info_stockfish = resultado.info
                if 'score' in info_stockfish:
                    self.estadisticas['evaluaciones'].append(str(info_stockfish['score']))
                if 'depth' in info_stockfish:
                    self.estadisticas['profundidades_alcanzadas'].append(info_stockfish['depth'])
            
            # Convertir movimiento a índice de acción
            movimientos_legales = list(tablero.legal_moves)
            try:
                indice_accion = movimientos_legales.index(mejor_movimiento)
            except ValueError:
                logger.error("Movimiento devuelto no es legal: %s", mejor_movimiento)
                return np.random.choice(len(acciones_legales))
            
            # Actualizar estadísticas
            tiempo_transcurrido = time.time() - inicio_tiempo
            self.estadisticas['jugadas_realizadas'] += 1
            self.estadisticas['tiempo_total'] += tiempo_transcurrido
            
            logger.info("Stockfish seleccionó: %s (%.3fs)", mejor_movimiento, tiempo_transcurrido)
            
            return indice_accion
            
        except Exception as e:
            logger.error("Error al seleccionar acción con Stockfish: %s", str(e))
            return np.random.choice(len(acciones_legales))
    
    def obtener_evaluacion(self, tablero: chess.Board, tiempo: float = 0.1) -> float:
        """
        Obtiene la evaluación numérica de una posición.
        
        Args:
            tablero: Tablero a evaluar
            tiempo: Tiempo de análisis
            
        Returns:
            Evaluación en centipeones (100 = 1 peón de ventaja)
        """
        try:
            if self.engine is None:
                return 0.0
            
            info = self.engine.analyse(tablero, chess.engine.Limit(time=tiempo))
            score = info.get('score')
            
            if score is None:
                return 0.0
            
            # Convertir score relativo al jugador actual
            score_relativo = score.relative
            
            # Si es mate, devolver valor grande
            if score_relativo.is_mate():
                mate_en = score_relativo.mate()
                return 10000.0 if mate_en > 0 else -10000.0
            
            # Devolver score en centipeones
            return float(score_relativo.score())
            
        except Exception as e:
            logger.error("Error al evaluar posición: %s", str(e))
            return 0.0
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Obtiene las estadísticas del agente.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = self.estadisticas.copy()
        if stats['jugadas_realizadas'] > 0:
            stats['tiempo_promedio'] = stats['tiempo_total'] / stats['jugadas_realizadas']
        else:
            stats['tiempo_promedio'] = 0.0
        
        if stats['profundidades_alcanzadas']:
            stats['profundidad_promedio'] = np.mean(stats['profundidades_alcanzadas'])
        else:
            stats['profundidad_promedio'] = 0
        
        return stats
    
    def reiniciar_estadisticas(self):
        """Reinicia las estadísticas del agente."""
        self.estadisticas = {
            'jugadas_realizadas': 0,
            'tiempo_total': 0.0,
            'evaluaciones': [],
            'profundidades_alcanzadas': []
        }
    
    def cerrar(self):
        """Cierra el motor de Stockfish."""
        if self.engine is not None:
            try:
                self.engine.quit()
                logger.info("Motor Stockfish cerrado correctamente")
            except Exception as e:
                logger.error("Error al cerrar Stockfish: %s", str(e))
            finally:
                self.engine = None
    
    def __del__(self):
        """Destructor para asegurar que el motor se cierre."""
        self.cerrar()
    
    def __str__(self) -> str:
        """Representación en string del agente."""
        return f"Stockfish(nivel={self.nivel_habilidad}, tiempo={self.tiempo_limite}s)"
    
    def __repr__(self) -> str:
        """Representación técnica del agente."""
        return (f"AgenteStockfish(nivel_habilidad={self.nivel_habilidad}, "
                f"profundidad={self.profundidad}, tiempo_limite={self.tiempo_limite})")
