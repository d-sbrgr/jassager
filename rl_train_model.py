import random
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from bots.rl_bots.util.utils import save_model, load_model
from jass.arena.arena import Arena
from jass.game.game_state import GameState
from bots.rl_bots.util.jassnet import JassNet
from bots.rl_bots.util.jassnet2 import JassNet2
from bots.rl_bots.rl_agent_cheating import RLAgentCheating
from bots.rl_bots.util.replay_buffer import ReplayBuffer
from bots.random_bot.full_random_cheating import RandomAgentCheating
from bots.heuristic_bots.full_heuristic_v2_cheating import FullHeuristicTableViewCheating

# Hyperparameters
learning_rate = 0.0001
gamma = 0.95
batch_size = 32
episodes = 2000
training_epochs = 50
max_buffer_size = 50000
model_path = "jass2_scrofa_v1.pth"
csv_file = "jass_data/rl_training_data/jass2_scrofaV1_vs_Mix_v1_data.csv"

# Initialize ReplayBuffer
replay_buffer = ReplayBuffer(capacity=max_buffer_size)

game_state = GameState()

# DEBUG flag
DEBUG = False

# Load or initialize model
if os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}")
    model = load_model(JassNet2, filepath=model_path)
else:
    print("No pretrained model found. Initializing a new model.")
    model = JassNet2(input_dim=629, action_dim=36)

# Initialize agent and optimizer
agent = RLAgentCheating(model)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# Initialize Arena
arena = Arena(nr_games_to_play=episodes, cheating_mode=True)

# Ensure my_team is an instance of RLAgentCheating
my_team = agent

# Load or initialize CSV
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
else:
    df = pd.DataFrame(columns=["episode", "team_points", "opponent_points", "win", "total_wins", "win_rate", "reward", "avg_reward", "policy_loss", "value_loss", "total_loss", "epsilon"])

# Define a function for sampling batches
def sample_batch(buffer, batch_size, recent=True):
    if recent:
        if len(buffer) < batch_size:
            batch = buffer.sample(len(buffer))  # Sample entire buffer if too small
        else:
            batch = buffer.sample_recent(batch_size)
    else:
        if len(buffer) < batch_size:
            batch = buffer.sample(len(buffer))  # Sample entire buffer if too small
        else:
            batch = buffer.sample(batch_size)

    if len(batch) == 0:  # Handle case where buffer has no samples
        return None
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.long),
        torch.tensor(np.array(rewards), dtype=torch.float32),
        torch.tensor(np.array(next_states), dtype=torch.float32),
        torch.tensor(np.array(dones), dtype=torch.float32),
    )


# Initialize variables to track wins
total_wins = 0

winrate = 0.0

# Training loop
for episode in tqdm(range(episodes), desc="Episodes"):
    print(f"Playing game {episode + 1}/{episodes}...")
    print(f"Current epsilon: {agent.epsilon}")

    # Reset agent state before each game
    # agent.reset()
    if episode == 1000:
        agent.epsilon = 0.5
        agent.epsilon_decay = 0.9995
    if (episode + 1) % 5 == 0:
        # Self-play: All players are RL agents
        arena.set_players(my_team, my_team, my_team, my_team)
        # print("Selfplay")
    else:
        opponent_team = RandomAgentCheating()
        # Regular games: Mix of RL agent and opponents
        if episode > 10:
            opponent_team = RandomAgentCheating()
            # if isinstance(opponent_team, RandomAgentCheating):
                # print(f"Opponent: RandomAgentCheating")
            if winrate > 70:
                opponent_team = FullHeuristicTableViewCheating()
                # if isinstance(opponent_team, RandomAgentCheating):
                    # print(f"Opponent: RandomAgentCheating")
                # if isinstance(opponent_team, FullHeuristicTableViewCheating):
                    # print(f"Opponent: FullHeuristicTableViewCheating")


        arena.set_players(my_team, opponent_team, my_team, opponent_team)
    print(f"Winrate: {winrate}")
    # print(f"Total wins: {total_wins}")

    # Play the game
    # try:
    arena.play_game(dealer=random.randint(0, 3))

    # Finalize the game to update terminal rewards
    agent.finalize_game(arena.get_observation())

    team_points_one_game = 0
    opponent_points_one_game = 0

    # Collect intermediate rewards and add to replay buffer
    for rl_agent in [arena.north, arena.south, arena.east, arena.west]:
        if isinstance(rl_agent, RLAgentCheating):
            for (state, action, reward, next_state, done) in rl_agent.experience_buffer:
                # Determine if the game was won
                obs_or_state = arena.get_observation()
                if obs_or_state.points[0] > obs_or_state.points[1]:
                    is_winning_game = True
                else:
                    is_winning_game = False
                team_points_one_game = obs_or_state.points[0]
                opponent_points_one_game = obs_or_state.points[1]

                # Push the transition with the winning/losing flag
                replay_buffer.push_with_prio((state, action, reward, next_state, done), is_winning_game)

            rl_agent.experience_buffer.clear()

    # except Exception as e:
    #     print(f"Error during game execution: {e}")
    #     continue

    # Train the model if enough samples in the buffer
    if len(replay_buffer) >= batch_size:
        dynamic_epochs = min(training_epochs, len(replay_buffer) // batch_size)
        for _ in range(dynamic_epochs):
            batch = sample_batch(replay_buffer, batch_size, recent=False)
            if batch is None:
                continue  # Skip if not enough samples
            states, actions, rewards, next_states, dones = batch

            # Forward pass
            policy, values = model(states)

            # Compute target values
            with torch.no_grad():
                _, next_values = model(next_states)
                target_values = rewards + gamma * next_values.squeeze() * (1 - dones)

            # Compute losses
            value_loss = torch.nn.functional.mse_loss(values.squeeze(), target_values)
            policy_loss = -torch.mean(policy.gather(1, actions.unsqueeze(1)).squeeze())
            total_loss = value_loss + 0.5 * policy_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Log metrics
    team_points = arena.points_team_0.sum()
    opponent_points = arena.points_team_1.sum()
    if is_winning_game:
        win = 1
    else:
        win = 0
    total_wins += win
    winrate = (total_wins / (episode + 1)) * 100

    print(f"Logged from rl_train_model.py: Team points: {team_points_one_game}, Opponent points: {opponent_points_one_game}")

    # Calculate average reward from replay buffer
    rewards = [transition[2] for transition in replay_buffer.buffer]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    new_row = {
        "episode": episode + 1,
        "team_points": team_points,
        "opponent_points": opponent_points,
        "win": win,
        "total_wins": total_wins,
        "win_rate": winrate,
        "rewards": sum(rewards),
        "avg_reward": avg_reward,
        "policy_loss": policy_loss.item() if 'policy_loss' in locals() else None,
        "value_loss": value_loss.item() if 'value_loss' in locals() else None,
        "total_loss": total_loss.item() if 'total_loss' in locals() else None,
        "epsilon": agent.epsilon
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Save final model and metrics
save_model(model, filepath=model_path)
df.to_csv(csv_file)
print("Training complete. Model and metrics saved.")
