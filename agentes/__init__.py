"""
Módulo agentes - Implementaciones de diferentes agentes de ajedrez.

Este módulo contiene las implementaciones de diferentes tipos de agentes:
- Minimax: Agente de búsqueda con poda alfa-beta
<<<<<<< Updated upstream
- MCTS: Agente Monte Carlo Tree Search  
- AlphaZero: Agente con MCTS + Redes Neuronales
- PPO: Agente Proximal Policy Optimization (futuro)
=======
- MCTS: Agente Monte Carlo Tree Search
- Stockfish: Motor de ajedrez UCI de clase mundial
- AlphaZero: MCTS + Redes Neuronales Profundas
- PPO: Agente Proximal Policy Optimization
>>>>>>> Stashed changes
"""

from .agente_minimax import AgenteMinimax
from .agente_mcts import AgenteMCTS
<<<<<<< Updated upstream
from .agente_alphazero import AgenteAlphaZero
=======
from .agente_stockfish import AgenteStockfish
#from .agente_dqn import AgenteDQN
from .agente_ppo import AgentePPO
>>>>>>> Stashed changes

__all__ = [
    'AgenteMinimax',
    'AgenteMCTS',
<<<<<<< Updated upstream
    'AgenteAlphaZero'
=======
    'AgenteStockfish',
    'AgenteDQN',
    'AgentePPO'
>>>>>>> Stashed changes
]