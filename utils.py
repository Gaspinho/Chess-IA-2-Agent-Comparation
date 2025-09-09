# Utilidades comunes para el proyecto de comparación de agentes de ajedrez

import os
import json
import pickle
import pandas as pd
import chess
import chess.engine
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EstadisticasPartida:
    """
    Clase para almacenar estadísticas de una partida.
    """
    def __init__(self):
        self.jugador_blanco = ""
        self.jugador_negro = ""
        self.resultado = ""  # "1-0", "0-1", "1/2-1/2"
        self.num_movimientos = 0
        self.tiempo_blanco = 0.0
        self.tiempo_negro = 0.0
        self.movimientos = []
        self.evaluaciones = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la estadística a diccionario."""
        return {
            'jugador_blanco': self.jugador_blanco,
            'jugador_negro': self.jugador_negro,
            'resultado': self.resultado,
            'num_movimientos': self.num_movimientos,
            'tiempo_blanco': self.tiempo_blanco,
            'tiempo_negro': self.tiempo_negro,
            'movimientos': self.movimientos,
            'evaluaciones': self.evaluaciones
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EstadisticasPartida':
        """Crea una instancia desde un diccionario."""
        stats = cls()
        for key, value in data.items():
            setattr(stats, key, value)
        return stats

def guardar_objeto(objeto: Any, ruta: str) -> None:
    """
    Guarda un objeto usando pickle.
    
    Args:
        objeto: Objeto a guardar
        ruta: Ruta donde guardar el archivo
    """
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'wb') as f:
            pickle.dump(objeto, f)
        logger.info(f"Objeto guardado en {ruta}")
    except Exception as e:
        logger.error(f"Error al guardar objeto en {ruta}: {e}")

def cargar_objeto(ruta: str) -> Any:
    """
    Carga un objeto usando pickle.
    
    Args:
        ruta: Ruta del archivo a cargar
        
    Returns:
        Objeto cargado o None si hay error
    """
    try:
        with open(ruta, 'rb') as f:
            objeto = pickle.load(f)
        logger.info(f"Objeto cargado desde {ruta}")
        return objeto
    except Exception as e:
        logger.error(f"Error al cargar objeto desde {ruta}: {e}")
        return None

def guardar_json(datos: Dict[str, Any], ruta: str) -> None:
    """
    Guarda datos en formato JSON.
    
    Args:
        datos: Diccionario con los datos
        ruta: Ruta donde guardar el archivo
    """
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON guardado en {ruta}")
    except Exception as e:
        logger.error(f"Error al guardar JSON en {ruta}: {e}")

def cargar_json(ruta: str) -> Optional[Dict[str, Any]]:
    """
    Carga datos desde un archivo JSON.
    
    Args:
        ruta: Ruta del archivo JSON
        
    Returns:
        Diccionario con los datos o None si hay error
    """
    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        logger.info(f"JSON cargado desde {ruta}")
        return datos
    except Exception as e:
        logger.error(f"Error al cargar JSON desde {ruta}: {e}")
        return None

def mover_a_notacion(movimiento: chess.Move, tablero: chess.Board) -> str:
    """
    Convierte un movimiento a notación algebraica estándar.
    
    Args:
        movimiento: Movimiento de chess
        tablero: Estado del tablero
        
    Returns:
        Movimiento en notación algebraica
    """
    return tablero.san(movimiento)

def fen_a_vector(fen: str) -> List[float]:
    """
    Convierte una posición FEN a un vector numérico.
    
    Args:
        fen: String FEN del tablero
        
    Returns:
        Vector numérico representando la posición
    """
    tablero = chess.Board(fen)
    vector = []
    
    # Convertir piezas a números (8x8 = 64 valores)
    mapa_piezas = {
        chess.PAWN: 1, chess.ROOK: 2, chess.KNIGHT: 3,
        chess.BISHOP: 4, chess.QUEEN: 5, chess.KING: 6
    }
    
    for square in chess.SQUARES:
        pieza = tablero.piece_at(square)
        if pieza is None:
            vector.append(0)
        else:
            valor = mapa_piezas[pieza.piece_type]
            if pieza.color == chess.BLACK:
                valor = -valor
            vector.append(valor)
    
    # Información adicional
    vector.append(1 if tablero.turn == chess.WHITE else -1)  # Turno
    vector.append(1 if tablero.has_kingside_castling_rights(chess.WHITE) else 0)
    vector.append(1 if tablero.has_queenside_castling_rights(chess.WHITE) else 0)
    vector.append(1 if tablero.has_kingside_castling_rights(chess.BLACK) else 0)
    vector.append(1 if tablero.has_queenside_castling_rights(chess.BLACK) else 0)
    vector.append(tablero.halfmove_clock / 100.0)  # Normalizado
    vector.append(tablero.fullmove_number / 100.0)  # Normalizado
    
    return vector

def calcular_elo(rating_a: float, rating_b: float, resultado: float, k: float = 32) -> tuple:
    """
    Calcula nuevos ratings ELO después de una partida.
    
    Args:
        rating_a: Rating actual del jugador A
        rating_b: Rating actual del jugador B
        resultado: Resultado para A (1=victoria, 0.5=empate, 0=derrota)
        k: Factor K de ELO
        
    Returns:
        Tuple con los nuevos ratings (nuevo_a, nuevo_b)
    """
    # Probabilidad esperada
    qa = 10 ** (rating_a / 400)
    qb = 10 ** (rating_b / 400)
    ea = qa / (qa + qb)
    eb = qb / (qa + qb)
    
    # Nuevos ratings
    nuevo_a = rating_a + k * (resultado - ea)
    nuevo_b = rating_b + k * ((1 - resultado) - eb)
    
    return nuevo_a, nuevo_b

def formatear_tiempo(segundos: float) -> str:
    """
    Formatea tiempo en segundos a formato legible.
    
    Args:
        segundos: Tiempo en segundos
        
    Returns:
        String formateado (ej: "2m 30s")
    """
    if segundos < 60:
        return f"{segundos:.1f}s"
    elif segundos < 3600:
        minutos = int(segundos // 60)
        segs = segundos % 60
        return f"{minutos}m {segs:.1f}s"
    else:
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        return f"{horas}h {minutos}m"

def validar_archivo_modelo(ruta: str) -> bool:
    """
    Valida si un archivo de modelo existe y es válido.
    
    Args:
        ruta: Ruta al archivo del modelo
        
    Returns:
        True si el archivo es válido
    """
    if not os.path.exists(ruta):
        return False
    
    try:
        # Intentar cargar para verificar integridad
        if ruta.endswith('.pkl'):
            cargar_objeto(ruta)
        elif ruta.endswith('.zip'):
            # Para modelos de stable-baselines3
            import zipfile
            with zipfile.ZipFile(ruta, 'r') as zf:
                zf.testzip()
        return True
    except Exception as e:
        logger.error(f"Archivo de modelo corrupto {ruta}: {e}")
        return False

class RegistroTorneos:
    """
    Clase para registrar y gestionar resultados de torneos.
    """
    def __init__(self):
        self.partidas = []
        self.ratings = {}
        
    def agregar_partida(self, stats: EstadisticasPartida):
        """Agrega una partida al registro."""
        self.partidas.append(stats)
        
        # Inicializar ratings si es necesario
        if stats.jugador_blanco not in self.ratings:
            self.ratings[stats.jugador_blanco] = 1500.0
        if stats.jugador_negro not in self.ratings:
            self.ratings[stats.jugador_negro] = 1500.0
        
        # Actualizar ratings ELO
        if stats.resultado == "1-0":
            resultado = 1.0
        elif stats.resultado == "0-1":
            resultado = 0.0
        else:
            resultado = 0.5
            
        nuevo_blanco, nuevo_negro = calcular_elo(
            self.ratings[stats.jugador_blanco],
            self.ratings[stats.jugador_negro],
            resultado
        )
        
        self.ratings[stats.jugador_blanco] = nuevo_blanco
        self.ratings[stats.jugador_negro] = nuevo_negro
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del torneo."""
        if not self.partidas:
            return {}
        
        # Crear DataFrame para análisis
        datos = []
        for partida in self.partidas:
            datos.append(partida.to_dict())
        
        df = pd.DataFrame(datos)
        
        # Calcular estadísticas por jugador
        jugadores = list(self.ratings.keys())
        stats = {}
        
        for jugador in jugadores:
            como_blanco = df[df['jugador_blanco'] == jugador]
            como_negro = df[df['jugador_negro'] == jugador]
            
            victorias = (
                len(como_blanco[como_blanco['resultado'] == '1-0']) +
                len(como_negro[como_negro['resultado'] == '0-1'])
            )
            
            empates = (
                len(como_blanco[como_blanco['resultado'] == '1/2-1/2']) +
                len(como_negro[como_negro['resultado'] == '1/2-1/2'])
            )
            
            derrotas = (
                len(como_blanco[como_blanco['resultado'] == '0-1']) +
                len(como_negro[como_negro['resultado'] == '1-0'])
            )
            
            total = victorias + empates + derrotas
            
            stats[jugador] = {
                'victorias': victorias,
                'empates': empates,
                'derrotas': derrotas,
                'total': total,
                'puntos': victorias + empates * 0.5,
                'porcentaje': (victorias + empates * 0.5) / max(total, 1) * 100,
                'rating_elo': self.ratings[jugador],
                'tiempo_promedio': (
                    como_blanco['tiempo_blanco'].mean() if len(como_blanco) > 0 else 0 +
                    como_negro['tiempo_negro'].mean() if len(como_negro) > 0 else 0
                ) / 2
            }
        
        return {
            'jugadores': stats,
            'total_partidas': len(self.partidas),
            'dataframe': df
        }
    
    def guardar(self, ruta: str):
        """Guarda el registro en un archivo."""
        datos = {
            'partidas': [p.to_dict() for p in self.partidas],
            'ratings': self.ratings
        }
        guardar_json(datos, ruta)
    
    @classmethod
    def cargar(cls, ruta: str) -> 'RegistroTorneos':
        """Carga un registro desde un archivo."""
        registro = cls()
        datos = cargar_json(ruta)
        
        if datos:
            registro.ratings = datos.get('ratings', {})
            for partida_data in datos.get('partidas', []):
                partida = EstadisticasPartida.from_dict(partida_data)
                registro.partidas.append(partida)
        
        return registro
