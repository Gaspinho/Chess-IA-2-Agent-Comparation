"""
Módulo agentes - Implementaciones de diferentes agentes de ajedrez.

Este módulo contiene las implementaciones de diferentes tipos de agentes:
- Minimax: Agente de búsqueda con poda alfa-beta
- MCTS: Agente Monte Carlo Tree Search
- Stockfish: Motor de ajedrez UCI de clase mundial
- AlphaZero: MCTS + Redes Neuronales Profundas
- PPO: Agente Proximal Policy Optimization
"""

from .agente_minimax import AgenteMinimax
from .agente_mcts import AgenteMCTS
from .agente_stockfish import AgenteStockfish
from .agente_alphazero import AgenteAlphaZero
from .agente_ppo import AgentePPO

__all__ = [
    'AgenteMinimax',
    'AgenteMCTS',
    'AgenteStockfish',
    'AgenteAlphaZero',
    'AgentePPO'
]