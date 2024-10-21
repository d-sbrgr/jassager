import numpy as np
import time

from jass.arena.arena import Arena

from bots import FullHeuristicEgocentric, FullMCTS, RandomAgent, CheatingMinimax

np.random.seed(0x2112)

MY_TEAM_AGENT_TYPE = FullMCTS
OPPONENT_TEAM_AGENT_TYPE = FullMCTS

for game, c_param in enumerate((1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4), 1):
    arena = Arena(nr_games_to_play=5)
    arena.set_players(
        MY_TEAM_AGENT_TYPE(c_param=c_param),
        OPPONENT_TEAM_AGENT_TYPE(iterations=1000),
        MY_TEAM_AGENT_TYPE(c_param=c_param),
        OPPONENT_TEAM_AGENT_TYPE(iterations=1000)
    )

    #print("DET: AVG |    MAX   |    MIN   ||--|| ALG: AVG |    MAX   |    MIN    ||--|| CDS: ")

    start = time.time()
    arena.play_all_games()
    stop = time.time()

    print(
        f"==========================\n"
        f"GAME: {game}\n"
        f"My team points: {arena.points_team_0.sum()}\n"
        f"Opponent team points: {arena.points_team_1.sum()}\n"
        f"Elapsed time: {stop - start:.4f} s\n"
        f"C_PARAM: {c_param:.1f}\n"
        f"==========================\n"
)
