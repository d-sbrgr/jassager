import random
import os
import torch
import numpy as np
import pandas as pd
from bots.rl_bots.util.utils import save_model, load_model
from jass.arena.arena import Arena
from bots.rl_bots.rl_agent import RLAgent
from bots.rl_bots.util.jassnet import JassNet
from bots.random_bot.full_random import RandomAgent
from bots.mcts_bots.full_mcts import FullMCTS

# Hyperparameters
learning_rate = 0.0001
gamma = 0.95  # Discount factor
batch_size = 32
episodes = 100  # Number of training games
training_epochs = 100
max_buffer_size = 50000  # Cap for the replay buffer
model_path = "trained_rl_model.pth"  # Path to the saved model
csv_file = "rl_training_data.csv"  # Path for logging training metrics

# Check if a saved model exists
if os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}")
    model = load_model(JassNet, filepath=model_path)
else:
    print("No pretrained model found. Initializing a new model.")
    model = JassNet(input_dim=481, action_dim=36)

agent = RLAgent(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the arena
arena = Arena(nr_games_to_play=episodes)  # Play one game per episode

# Initialize or load the CSV for metrics
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
else:
    df = pd.DataFrame(columns=["team_points", "opponent_points", "win_rate", "avg_reward"])

# Replay buffer
replay_buffer = []

# Function to sample experiences from the buffer
def sample_batch(buffer, batch_size):
    batch = random.sample(buffer, min(len(buffer), batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.long),
        torch.tensor(np.array(rewards), dtype=torch.float32),
        torch.tensor(np.array(next_states), dtype=torch.float32),
        torch.tensor(np.array(dones), dtype=torch.float32),
    )

# Training loop
for episode in range(episodes):
    print(f"Playing game {episode + 1}/{episodes}...")

    # Alternate between self-play and opponents
    if episode % 3 == 0:
        arena.set_players(agent, agent, agent, agent)  # Self-play
    else:
        opponent = RandomAgent() if episode % 2 == 0 else FullMCTS()
        arena.set_players(agent, opponent, agent, opponent)

    # Play one game
    arena.play_game(dealer=random.randint(0, 3))

    # Collect experiences and clear agent buffers
    for rl_agent in [arena.north, arena.south, arena.east, arena.west]:
        if isinstance(rl_agent, RLAgent):
            replay_buffer.extend(rl_agent.experience_buffer)
            rl_agent.experience_buffer.clear()

    # Limit the replay buffer size
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[-max_buffer_size:]

    # Train the model if sufficient data is in the buffer
    if len(replay_buffer) >= batch_size:
        dynamic_epochs = min(training_epochs, len(replay_buffer) // batch_size)
        for _ in range(dynamic_epochs):
            states, actions, rewards, next_states, dones = sample_batch(replay_buffer, batch_size)

            # Forward pass
            policy, values = model(states)
            policy_values = policy.gather(1, actions.unsqueeze(1)).squeeze()

            # Compute target values
            with torch.no_grad():
                _, next_values = model(next_states)
                target_values = rewards + gamma * next_values.squeeze() * (1 - dones)

            # Compute losses
            value_loss = torch.nn.functional.mse_loss(values.squeeze(), target_values)
            policy_loss = -torch.mean(policy_values)
            loss = value_loss + 0.5 * policy_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate episode metrics
    rewards = [transition[2] for transition in replay_buffer]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    team_points = arena.points_team_0.sum()
    opponent_points = arena.points_team_1.sum()
    win_rate = team_points / (team_points + opponent_points) * 100 if (team_points + opponent_points) > 0 else 0

    # Log metrics to the CSV
    new_row = {"team_points": team_points, "opponent_points": opponent_points, "win_rate": win_rate, "avg_reward": avg_reward}
    if any(new_row.values()):
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save checkpoints
    if episode % 10 == 0:
        save_model(model, filepath=f"trained_rl_model_checkpoint_{episode}.pth")
        print(f"Checkpoint saved at episode {episode}")

    print(f"Episode {episode + 1}/{episodes}: "
          f"Win Rate = {win_rate:.2f}%, "
          f"Average Reward = {avg_reward:.2f}, "
          f"Replay Buffer Size = {len(replay_buffer)}")

# Save the final model and CSV
save_model(model, filepath=model_path)
df.to_csv(csv_file)
print("Training complete. Model and metrics saved.")
