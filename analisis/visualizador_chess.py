"""
Visualizador gráfico para partidas de ajedrez usando pygame.

Este módulo proporciona una interfaz gráfica para mostrar las partidas
de ajedrez en tiempo real con información sobre los agentes y estadísticas.
"""

import pygame
import chess
import chess.svg
import numpy as np
import sys
import time
from typing import Optional, Dict, Any, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuraciones de colores
COLORES = {
    'CASILLA_CLARA': (240, 217, 181),
    'CASILLA_OSCURA': (181, 136, 99),
    'BORDE': (139, 69, 19),
    'FONDO': (50, 50, 50),
    'TEXTO': (255, 255, 255),
    'HIGHLIGHT_ULTIMA_JUGADA': (255, 255, 0, 100),
    'HIGHLIGHT_CHECK': (255, 0, 0, 100),
    'PANEL_INFO': (30, 30, 30),
    'VERDE': (0, 200, 0),
    'ROJO': (200, 0, 0),
    'AZUL': (0, 100, 200)
}


class VisualizadorAjedrez:
    """
    Visualizador gráfico para partidas de ajedrez.
    """
    
    def __init__(self, tamaño_casilla: int = 64):
        """
        Inicializa el visualizador.
        
        Args:
            tamaño_casilla: Tamaño en píxeles de cada casilla
        """
        self.tamaño_casilla = tamaño_casilla
        self.tamaño_tablero = tamaño_casilla * 8
        self.ancho_panel = 300
        self.altura_ventana = self.tamaño_tablero + 100
        self.ancho_ventana = self.tamaño_tablero + self.ancho_panel
        
        # Inicializar pygame
        pygame.init()
        self.pantalla = pygame.display.set_mode((self.ancho_ventana, self.altura_ventana))
        pygame.display.set_caption("Chess AI Visualizer - Minimax vs MCTS")
        
        # Fuentes
        self.fuente_grande = pygame.font.Font(None, 24)
        self.fuente_mediana = pygame.font.Font(None, 20)
        self.fuente_pequeña = pygame.font.Font(None, 16)
        
        # Estado del juego
        self.tablero = chess.Board()
        self.ultima_jugada = None
        self.info_agentes = {}
        self.estadisticas_partida = {}
        self.reloj = pygame.time.Clock()
        
        # Cargar imágenes de piezas (usando caracteres Unicode por ahora)
        self.simbolos_piezas = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        logger.info("Visualizador inicializado - Tamaño ventana: %dx%d", 
                   self.ancho_ventana, self.altura_ventana)
    
    def convertir_posicion_a_coordenadas(self, fila: int, columna: int) -> Tuple[int, int]:
        """
        Convierte posición del tablero a coordenadas de píxeles.
        
        Args:
            fila: Fila del tablero (0-7)
            columna: Columna del tablero (0-7)
            
        Returns:
            Tupla (x, y) de coordenadas de píxeles
        """
        x = columna * self.tamaño_casilla
        y = (7 - fila) * self.tamaño_casilla  # Invertir Y para mostrar correctamente
        return x, y
    
    def dibujar_tablero(self):
        """Dibuja el tablero de ajedrez."""
        for fila in range(8):
            for columna in range(8):
                x, y = self.convertir_posicion_a_coordenadas(fila, columna)
                
                # Color de la casilla
                if (fila + columna) % 2 == 0:
                    color = COLORES['CASILLA_CLARA']
                else:
                    color = COLORES['CASILLA_OSCURA']
                
                pygame.draw.rect(self.pantalla, color, 
                               (x, y, self.tamaño_casilla, self.tamaño_casilla))
    
    def destacar_ultima_jugada(self):
        """Destaca la última jugada realizada."""
        if self.ultima_jugada:
            # Destacar casilla origen
            from_square = self.ultima_jugada.from_square
            fila_origen = chess.square_rank(from_square)
            columna_origen = chess.square_file(from_square)
            x_origen, y_origen = self.convertir_posicion_a_coordenadas(fila_origen, columna_origen)
            
            superficie_highlight = pygame.Surface((self.tamaño_casilla, self.tamaño_casilla))
            superficie_highlight.set_alpha(100)
            superficie_highlight.fill(COLORES['HIGHLIGHT_ULTIMA_JUGADA'][:3])
            self.pantalla.blit(superficie_highlight, (x_origen, y_origen))
            
            # Destacar casilla destino
            to_square = self.ultima_jugada.to_square
            fila_destino = chess.square_rank(to_square)
            columna_destino = chess.square_file(to_square)
            x_destino, y_destino = self.convertir_posicion_a_coordenadas(fila_destino, columna_destino)
            
            self.pantalla.blit(superficie_highlight, (x_destino, y_destino))
    
    def destacar_jaque(self):
        """Destaca la casilla del rey si está en jaque."""
        if self.tablero.is_check():
            rey_en_jaque = self.tablero.king(self.tablero.turn)
            if rey_en_jaque:
                fila = chess.square_rank(rey_en_jaque)
                columna = chess.square_file(rey_en_jaque)
                x, y = self.convertir_posicion_a_coordenadas(fila, columna)
                
                superficie_check = pygame.Surface((self.tamaño_casilla, self.tamaño_casilla))
                superficie_check.set_alpha(150)
                superficie_check.fill(COLORES['HIGHLIGHT_CHECK'][:3])
                self.pantalla.blit(superficie_check, (x, y))
    
    def dibujar_piezas(self):
        """Dibuja las piezas en el tablero."""
        for fila in range(8):
            for columna in range(8):
                square = chess.square(columna, fila)
                pieza = self.tablero.piece_at(square)
                
                if pieza:
                    x, y = self.convertir_posicion_a_coordenadas(fila, columna)
                    simbolo = self.simbolos_piezas.get(pieza.symbol(), '?')
                    
                    # Renderizar el símbolo de la pieza
                    texto_pieza = self.fuente_grande.render(simbolo, True, COLORES['TEXTO'])
                    rect_texto = texto_pieza.get_rect()
                    rect_texto.center = (x + self.tamaño_casilla // 2, y + self.tamaño_casilla // 2)
                    self.pantalla.blit(texto_pieza, rect_texto)
    
    def dibujar_panel_informacion(self):
        """Dibuja el panel de información lateral."""
        x_panel = self.tamaño_tablero + 10
        y_panel = 10
        
        # Fondo del panel
        pygame.draw.rect(self.pantalla, COLORES['PANEL_INFO'], 
                        (x_panel - 5, y_panel - 5, self.ancho_panel - 10, self.altura_ventana - 20))
        
        # Información de los agentes
        y_actual = y_panel
        
        # Título
        titulo = self.fuente_grande.render("MINIMAX vs MCTS", True, COLORES['TEXTO'])
        self.pantalla.blit(titulo, (x_panel, y_actual))
        y_actual += 40
        
        # Información del turno actual
        turno = "Blancas" if self.tablero.turn else "Negras"
        agente_actual = self.info_agentes.get('agente_blancas', 'Minimax') if self.tablero.turn else self.info_agentes.get('agente_negras', 'MCTS')
        
        texto_turno = self.fuente_mediana.render(f"Turno: {turno}", True, COLORES['TEXTO'])
        self.pantalla.blit(texto_turno, (x_panel, y_actual))
        y_actual += 25
        
        texto_agente = self.fuente_mediana.render(f"Agente: {agente_actual}", True, COLORES['VERDE'])
        self.pantalla.blit(texto_agente, (x_panel, y_actual))
        y_actual += 35
        
        # Estado del juego
        if self.tablero.is_checkmate():
            estado = "¡JAQUE MATE!"
            color_estado = COLORES['ROJO']
        elif self.tablero.is_stalemate():
            estado = "TABLAS (Ahogado)"
            color_estado = COLORES['AZUL']
        elif self.tablero.is_check():
            estado = "JAQUE"
            color_estado = COLORES['ROJO']
        else:
            estado = "En progreso"
            color_estado = COLORES['VERDE']
        
        texto_estado = self.fuente_mediana.render(f"Estado: {estado}", True, color_estado)
        self.pantalla.blit(texto_estado, (x_panel, y_actual))
        y_actual += 35
        
        # Estadísticas de la partida
        num_jugadas = len(self.tablero.move_stack)
        texto_jugadas = self.fuente_pequeña.render(f"Jugadas: {num_jugadas}", True, COLORES['TEXTO'])
        self.pantalla.blit(texto_jugadas, (x_panel, y_actual))
        y_actual += 25
        
        # Información adicional de estadísticas
        for clave, valor in self.estadisticas_partida.items():
            if isinstance(valor, float):
                texto_valor = f"{clave}: {valor:.2f}"
            else:
                texto_valor = f"{clave}: {valor}"
            
            texto_estadistica = self.fuente_pequeña.render(texto_valor, True, COLORES['TEXTO'])
            self.pantalla.blit(texto_estadistica, (x_panel, y_actual))
            y_actual += 20
        
        # Última jugada
        if self.ultima_jugada:
            texto_ultima = self.fuente_pequeña.render(
                f"Última: {self.ultima_jugada.uci()}", True, COLORES['TEXTO'])
            self.pantalla.blit(texto_ultima, (x_panel, y_actual))
            y_actual += 25
        
        # Instrucciones
        y_actual += 20
        instrucciones = [
            "ESC - Salir",
            "SPACE - Pausar",
            "R - Reiniciar"
        ]
        
        for instruccion in instrucciones:
            texto_inst = self.fuente_pequeña.render(instruccion, True, COLORES['TEXTO'])
            self.pantalla.blit(texto_inst, (x_panel, y_actual))
            y_actual += 18
    
    def actualizar_tablero(self, tablero: chess.Board, ultima_jugada: Optional[chess.Move] = None):
        """
        Actualiza el estado del tablero mostrado.
        
        Args:
            tablero: Nuevo estado del tablero
            ultima_jugada: Última jugada realizada
        """
        self.tablero = tablero.copy()
        self.ultima_jugada = ultima_jugada
    
    def actualizar_info_agentes(self, info_agentes: Dict[str, Any]):
        """
        Actualiza la información de los agentes.
        
        Args:
            info_agentes: Diccionario con información de los agentes
        """
        self.info_agentes.update(info_agentes)
    
    def actualizar_estadisticas(self, estadisticas: Dict[str, Any]):
        """
        Actualiza las estadísticas de la partida.
        
        Args:
            estadisticas: Diccionario con estadísticas
        """
        self.estadisticas_partida.update(estadisticas)
    
    def renderizar(self):
        """Renderiza la pantalla completa."""
        # Limpiar pantalla
        self.pantalla.fill(COLORES['FONDO'])
        
        # Dibujar elementos
        self.dibujar_tablero()
        self.destacar_ultima_jugada()
        self.destacar_jaque()
        self.dibujar_piezas()
        self.dibujar_panel_informacion()
        
        # Actualizar display
        pygame.display.flip()
    
    def manejar_eventos(self) -> bool:
        """
        Maneja los eventos de pygame.
        
        Returns:
            True si debe continuar, False si debe salir
        """
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    return False
                elif evento.key == pygame.K_SPACE:
                    # Pausar/reanudar
                    logger.info("Pausa activada - presiona SPACE para continuar")
                    while True:
                        for sub_evento in pygame.event.get():
                            if sub_evento.type == pygame.QUIT:
                                return False
                            elif sub_evento.type == pygame.KEYDOWN:
                                if sub_evento.key == pygame.K_SPACE:
                                    return True
                                elif sub_evento.key == pygame.K_ESCAPE:
                                    return False
                        self.reloj.tick(30)
                elif evento.key == pygame.K_r:
                    # Reiniciar (implementar si es necesario)
                    logger.info("Reinicio solicitado")
        
        return True
    
    def cerrar(self):
        """Cierra el visualizador."""
        pygame.quit()
        logger.info("Visualizador cerrado")
    
    def esperar(self, milisegundos: int = 1000):
        """
        Espera un tiempo determinado manteniendo la interfaz responsiva.
        
        Args:
            milisegundos: Tiempo a esperar en milisegundos
        """
        tiempo_inicio = pygame.time.get_ticks()
        while pygame.time.get_ticks() - tiempo_inicio < milisegundos:
            if not self.manejar_eventos():
                return False
            self.reloj.tick(60)
        return True