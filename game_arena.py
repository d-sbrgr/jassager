import numpy as np
import time

from jass.arena.arena import Arena

from bots import *

np.random.seed(0xb48a)

MY_TEAM_AGENT_TYPE = MCTSCNNRollout
OPPONENT_TEAM_AGENT_TYPE = RandomAgent


for c_param in (1.0, 1.3):
    arena = Arena(nr_games_to_play=10)
    arena.set_players(
        MY_TEAM_AGENT_TYPE(c_param),
        OPPONENT_TEAM_AGENT_TYPE(),
        MY_TEAM_AGENT_TYPE(c_param),
        OPPONENT_TEAM_AGENT_TYPE()
    )

    start = time.time()
    arena.play_all_games()
    stop = time.time()

    print(
        f"==========================\n"
        f"My team points: {arena.points_team_0.sum()}\n"
        f"Opponent team points: {arena.points_team_1.sum()}\n"
        f"Elapsed time: {stop - start:.4f} s\n"
        f"C: {c_param}\n"
        f"==========================\n"
    )
