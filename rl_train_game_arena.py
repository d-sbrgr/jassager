import numpy as np
import time

import pandas as pd
import os
import torch

from jass.arena.arena import Arena
from bots.random_bot.full_random import RandomAgent
from bots.rl_bots.rl_agent import RLAgent
from bots.mcts_bots.full_mcts import FullMCTS
from bots.mcts_bots.heuristic_trump_mcts_play import HeuristicTrumpMCTSPlay
from bots.random_bot.full_random import RandomAgent
from bots.rl_bots.util.jassnet import JassNet
from bots.rl_bots.util.utils import load_model

np.random.seed(0xb48a)

# Load the trained model
model = load_model(JassNet, filepath="jass_scrofa_v5.pth")

# Create the RL agent with the loaded model
rl_agent = RLAgent(model)

nr_of_games = 10

# Create the arena
arena = Arena(nr_games_to_play=nr_of_games, cheating_mode=False)

# Set players directly
arena.set_players(
    rl_agent,                           # RLAgent for North
    FullMCTS(),                         # RandomAgent for East
    rl_agent,                           # RLAgent for South
    FullMCTS()                          # RandomAgent for West
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

# Define the CSV file path
csv_file = "jass_data/rl_training_data/test_game_arena_jass_scrofa_vs_heuristicTrumpMCTSPlay_v1.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0)  # Load existing DataFrame
else:
    df = pd.DataFrame(columns=["team_points", "opponent_points", "elapsed_time", "win", "winrate"])  # New DataFrame

total_wins = 0
team_points = arena.points_team_0.sum()
opponent_points = arena.points_team_1.sum()
elapsed_time = stop - start
win = 1 if team_points > opponent_points else 0
total_wins += win
winrate = (total_wins / (nr_of_games + 1)) * 100


new_row = {"team_points": team_points, "opponent_points": opponent_points, "elapsed_time": elapsed_time, "win": win, "winrate": winrate}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

df.to_csv(csv_file)

