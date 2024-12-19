import random
import os
import torch
import numpy as np
from jass.arena.arena import Arena
from bots.rl_bots.rl_agent import RLAgent
from bots.rl_bots.util.utils import load_model
from bots.rl_bots.util.jassnet import JassNet
from bots.random_bot.full_random import RandomAgent
from bots.heuristic_bots.full_heuristic_v2 import FullHeuristicTableView
from bots.mcts_bots.full_mcts import FullMCTS
import pandas as pd

# Hyperparameters
test_episodes = 10  # Number of test games
model_path = "models/rl_models/jass_scrofa_v2.pth"  # Path to the trained model
csv_test_file = "jass_data/rl_training_data/jass_test_metrics.csv"  # File to log test metrics

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
model = load_model(JassNet, filepath=model_path)

# Initialize Arena in non-cheating mode for testing
test_arena = Arena(nr_games_to_play=test_episodes, cheating_mode=False)

# Create RLAgent for testing (imperfect info)
test_agent = RLAgent(model, epsilon=0.0, epsilon_decay=0, min_epsilon=0)  # Fully exploit during testing
test_arena.set_players(test_agent, FullMCTS(), test_agent, FullMCTS())

# Initialize or load test metrics
if os.path.exists(csv_test_file):
    test_df = pd.read_csv(csv_test_file, index_col=0)
else:
    test_df = pd.DataFrame(columns=["episode", "team_points", "opponent_points", "win", "win_rate"])

# Initialize variables to track wins
total_wins = 0

# Play test games
for episode in range(test_episodes):
    print(f"Testing game {episode + 1}/{test_episodes}...")
    try:
        test_arena.play_game(dealer=random.randint(0, 3))
    except Exception as e:
        print(f"Error during test game execution: {e}")
        continue

    # Calculate game metrics
    team_points = test_arena.points_team_0.sum()
    opponent_points = test_arena.points_team_1.sum()
    win = 1 if team_points > opponent_points else 0
    total_wins += win
    win_rate = (total_wins / (episode + 1)) * 100  # Dynamic winrate calculation

    # Log results
    new_row = {
        "episode": episode + 1,
        "team_points": team_points,
        "opponent_points": opponent_points,
        "win": win,
        "win_rate": win_rate,
    }
    test_df = pd.concat([test_df, pd.DataFrame([new_row])], ignore_index=True)

# Save test metrics
test_df.to_csv(csv_test_file, index=False)
print(f"Testing complete. Metrics saved to {csv_test_file}.")
