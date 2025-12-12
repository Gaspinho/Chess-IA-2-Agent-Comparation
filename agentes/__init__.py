"""
Módulo agentes - Implementaciones de diferentes agentes de ajedrez.

Este módulo contiene las implementaciones de diferentes tipos de agentes:
- Minimax: Agente de búsqueda con poda alfa-beta
- MCTS: Agente Monte Carlo Tree Search  
- AlphaZero: Agente con MCTS + Redes Neuronales
- PPO: Agente Proximal Policy Optimization (futuro)
"""

from .agente_minimax import AgenteMinimax
from .agente_mcts import AgenteMCTS
from .agente_alphazero import AgenteAlphaZero

__all__ = [
    'AgenteMinimax',
    'AgenteMCTS',
    'AgenteAlphaZero'
]