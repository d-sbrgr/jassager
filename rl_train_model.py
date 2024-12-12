import random
import os
import torch
import numpy as np
import pandas as pd
from bots.rl_bots.util.utils import save_model, load_model
from jass.arena.arena import Arena
from bots.rl_bots.util.jassnet import JassNet
from bots.cheating.cheating_agents import RLAgentCheating, HeuristicAgentCheating, RandomAgentCheating

# Hyperparameters
learning_rate = 0.001
gamma = 0.95
batch_size = 32
episodes = 2000
training_epochs = 20
max_buffer_size = 50000
model_path = "jass_scrofa_v1.pth"
csv_file = "jass_scrofa_v1_data.csv"

# Load or initialize model
if os.path.exists(model_path):
    print(f"Loading pretrained model from {model_path}")
    model = load_model(JassNet, filepath=model_path)
else:
    print("No pretrained model found. Initializing a new model.")
    model = JassNet(input_dim=481, action_dim=36)

# Initialize agent and optimizer
agent = RLAgentCheating(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize Arena
arena = Arena(nr_games_to_play=episodes, cheating_mode=True)

# Load or initialize CSV
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
else:
    df = pd.DataFrame(columns=["episode", "team_points", "opponent_points", "win_rate", "avg_reward", "policy_loss", "value_loss", "total_loss"])

# Replay buffer
replay_buffer = []

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

# Initialize variables to track wins
total_wins = 0

# Training loop
for episode in range(episodes):
    print(f"Playing game {episode + 1}/{episodes}...")

    # Reset epsilon periodically
    if episode % 50 == 0 and episode != 0:
        agent.epsilon = 0.5  # Reset exploration
        print(f"Epsilon reset at episode {episode} to {agent.epsilon:.2f}")

    arena.set_players(agent, agent, agent, agent)  # Self-play

    # # Set opponents
    # if episode % 3 == 0:
    #     arena.set_players(agent, agent, agent, agent)  # Self-play
    # else:
    #     opponent = HeuristicAgentCheating() if episode % 5 == 0 else RandomAgentCheating()
    #     arena.set_players(agent, opponent, agent, opponent)

    # Play a game
    try:
        arena.play_game(dealer=random.randint(0, 3))
    except Exception as e:
        print(f"Error during game execution: {e}")
        continue

    # Collect experiences and clear agent buffers
    for rl_agent in [arena.north, arena.south, arena.east, arena.west]:
        if isinstance(rl_agent, RLAgentCheating):
            replay_buffer.extend(rl_agent.experience_buffer)
            rl_agent.experience_buffer.clear()

    # Cap replay buffer size
    if len(replay_buffer) > max_buffer_size:
        replay_buffer = replay_buffer[-max_buffer_size:]

    # Train the model
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
            total_loss = value_loss + 0.5 * policy_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Log metrics
    team_points = arena.points_team_0.sum()
    opponent_points = arena.points_team_1.sum()
    win = 1 if team_points > opponent_points else 0
    total_wins += win
    winrate = (total_wins / (episode + 1)) * 100

    avg_reward = sum([transition[2] for transition in replay_buffer]) / len(replay_buffer) if replay_buffer else 0

    new_row = {
        "episode": episode + 1,
        "team_points": team_points,
        "opponent_points": opponent_points,
        "win_rate": winrate,
        "win": win,
        "avg_reward": avg_reward,
        "policy_loss": policy_loss.item() if 'policy_loss' in locals() else None,
        "value_loss": value_loss.item() if 'value_loss' in locals() else None,
        "total_loss": total_loss.item() if 'total_loss' in locals() else None,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # # Save checkpoint
    # if episode % 10 == 0:
    #     save_model(model, filepath=f"trained_rl_model_checkpoint_{episode}.pth")
    #     print(f"Checkpoint saved at episode {episode}")

# Save final model and metrics
save_model(model, filepath=model_path)
df.to_csv(csv_file)
print("Training complete. Model and metrics saved.")