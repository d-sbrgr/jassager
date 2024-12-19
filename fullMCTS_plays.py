import random
import os
import pandas as pd
from jass.arena.arena import Arena
from jass.game.game_observation import GameObservation
from bots.mcts_bots.full_mcts_rl_master import FullMCTS

# Hyperparameters
num_games = 20  # Number of games to simulate
output_file = "jass_data/rl_training_data/full_mcts_experience.csv"  # File to save experience data

# Initialize Arena and FullMCTS agent
arena = Arena(nr_games_to_play=num_games, cheating_mode=False)
mcts_agent = FullMCTS()

# Set FullMCTS agent for all players
arena.set_players(mcts_agent, mcts_agent, mcts_agent, mcts_agent)

# Initialize storage for experience
experience_data = []

# Play games and collect experience
for game_id in range(num_games):
    print(f"Simulating game {game_id + 1}/{num_games}...")
    try:
        # Play one game
        arena.play_game(dealer=random.randint(0, 3))

        # Collect experience from each FullMCTS agent
        for agent in [arena.north, arena.south, arena.east, arena.west]:
            if isinstance(agent, FullMCTS) and agent.experience_buffer:
                experience_data.extend(agent.experience_buffer)
                agent.clear_experience_buffer()  # Clear buffer after saving

        print(f"Game {game_id + 1} complete.")

    except Exception as e:
        print(f"Error during game simulation: {e}")
        continue

# Save experience to CSV
if experience_data:
    experience_df = pd.DataFrame(experience_data)
    experience_df.to_csv(output_file, index=False)
    print(f"Experience data saved to {output_file}.")
else:
    print("No experience data collected.")
