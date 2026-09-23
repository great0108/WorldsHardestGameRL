from .core import WorldsHardestGameCore
from .game_data import GameData, COIN_COUNTS

__all__ = ["WorldsHardestGameEnv", "WorldsHardestGameCore", "GameData", "COIN_COUNTS"]

def __getattr__(name):
    if name == "WorldsHardestGameEnv":
        from .env import WorldsHardestGameEnv
        return WorldsHardestGameEnv
    raise AttributeError(name)
