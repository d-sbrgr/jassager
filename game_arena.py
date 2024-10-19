import numpy as np
import time

from jass.arena.arena import Arena

from bots import FullHeuristicEgocentric, FullMCTS, RandomAgent, CheatingMinimax

np.random.seed(0x2525)

MY_TEAM_AGENT_TYPE = FullMCTS
OPPONENT_TEAM_AGENT_TYPE = FullMCTS

arena = Arena(nr_games_to_play=2)
arena.set_players(
    MY_TEAM_AGENT_TYPE(iterations=100),
    OPPONENT_TEAM_AGENT_TYPE(iterations=1000),
    MY_TEAM_AGENT_TYPE(iterations=100),
    OPPONENT_TEAM_AGENT_TYPE(iterations=1000)
)

print("DET: AVG |    MAX   |    MIN   ||--|| ALG: AVG |    MAX   |    MIN    ||--|| CDS: ")

start = time.time()
arena.play_all_games()
stop = time.time()

print(
    f"==========================\n"
    f"My team points: {arena.points_team_0.sum()}\n"
    f"Opponent team points: {arena.points_team_1.sum()}\n"
    f"Elapsed time: {stop - start:.4f} s\n"
    f"==========================\n"
)
