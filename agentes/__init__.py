"""
Módulo agentes - Implementaciones de diferentes agentes de ajedrez.

Este módulo contiene las implementaciones de cuatro tipos de agentes:
- Minimax: Agente de búsqueda con poda alfa-beta
- MCTS: Agente Monte Carlo Tree Search  
- DQN: Agente Deep Q-Network
- PPO: Agente Proximal Policy Optimization
"""

from .agente_minimax import AgenteMinimax
from .agente_mcts import AgenteMCTS
from .agente_dqn import AgenteDQN
from .agente_ppo import AgentePPO

__all__ = [
    'AgenteMinimax',
    'AgenteMCTS', 
    'AgenteDQN',
    'AgentePPO'
]