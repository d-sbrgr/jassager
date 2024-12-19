import numpy as np
import torch

from jass.agents.agent_cheating import AgentCheating
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list

from bots.rl_bots.util.reward_system import calculate_rewards_state
from bots.rl_bots.util.encode_game_state import encode_game_state
from bots.rl_bots.util.utils import validate_game_state

# DEBUG flag
DEBUG = False

# Log flag
log = False

# immediate reward
immediate = True

if DEBUG:
    print(" ===== cheating_agent.py in DEBUG mode =====")

class RLAgentCheating(AgentCheating):
    def __init__(self, model, epsilon=1, epsilon_decay=0.99995, min_epsilon=0.1):
        """
        Initialize the RLAgentCheating.

        Args:
            model: Neural network model for policy and value predictions.
            epsilon: Initial exploration rate for epsilon-greedy policy.
            epsilon_decay: Decay factor for epsilon.
            min_epsilon: Minimum value for epsilon.
        """
        self.model = model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.experience_buffer = []
        self.previous_state = None
        self.previous_action = None
        self._rule = RuleSchieber()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"Current epsilon: {self.epsilon}")

    def action_trump(self, state: GameState) -> int:
        if DEBUG:
            print(f"DEBUG: Calling action_trump, Trump: {state.trump}")
        encoded_state = encode_game_state(state)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

        # Predict trump probabilities using the model
        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        # Valid trump actions
        valid_trumps = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]

        if np.random.rand() < self.epsilon:
            # Explore: Choose a random valid trump
            trump_action = np.random.choice(valid_trumps)
            if log:
                print(f"Exploring: Random trump action selected: {trump_action}")
        else:
            # Exploit: Choose trump with the highest predicted probability
            trump_action = max(valid_trumps, key=lambda t: policy[0, t].item() if 0 <= t < 6 else float('-inf'))


        # Calculate reward for the chosen trump
        reward = calculate_rewards_state(state, immediate=True, action=trump_action)
        if DEBUG:
            print(f"DEBUG: Reward for selected trump {trump_action}: {reward}")

        # Store the transition (if this is not the first action)
        if self.previous_state is not None:
            done = False  # Trump selection is never the end of the game
            transition = (self.previous_state, self.previous_action, reward, encoded_state, done)
            if log:
                print(
                    f"Storing trump transition - Action: {self.previous_action}, Reward: {reward}, "
                    f"State size: {len(self.previous_state)}, Next State size: {len(encoded_state)}"
                )
            self.experience_buffer.append(transition)

        # Update previous state and action
        self.previous_state = encoded_state
        self.previous_action = trump_action

        valid_trumps = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]
        assert trump_action in valid_trumps, f"Invalid trump action {trump_action}, valid trumps: {valid_trumps}"
        assert 0 <= trump_action < 6, f"Trump action {trump_action} is out of bounds"

        return trump_action

    def action_play_card(self, state: GameState) -> int:
        validate_game_state(state)
        if DEBUG:
            print(f"DEBUG: Calling action_play_card, Cards played: {state.nr_played_cards}")

        encoded_state = encode_game_state(state)
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_state(state)
        )

        if len(valid_moves) == 1:
            action = valid_moves[0]
            if DEBUG:
                print(f"DEBUG: Single valid move available: {action}")
        else:
            state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
            if np.random.rand() < self.epsilon:
                action = np.random.choice(valid_moves)
            else:
                with torch.no_grad():
                    policy_logits, _ = self.model(state_tensor)
                # Ensure only valid actions are considered
                policy_logits[:, [i for i in range(36) if i not in valid_moves]] = float('-inf')
                action_probabilities = torch.softmax(policy_logits, dim=-1).squeeze()
                action = torch.argmax(action_probabilities).item()
                if DEBUG:
                    print(f"DEBUG: Exploitation: Selected action {action} with probabilities {action_probabilities}")

        # Calculate reward for the current action
        reward = calculate_rewards_state(state, immediate=True, action=action)
        if log:
            print(f"reward: {reward}, for action: {action}")

        # Store transition
        if self.previous_state is not None:
            done = state.nr_played_cards == 36
            transition = (self.previous_state, self.previous_action, reward, encoded_state, done)
            self.experience_buffer.append(transition)
            if log:
                print(f"Storing transition - Action: {self.previous_action}, Reward: {reward}, Done: {done}")

        self.previous_state = encoded_state
        self.previous_action = action

        assert action in valid_moves, f"Invalid action chosen: {action}, valid moves: {valid_moves}"

        return action

    def finalize_game(self, state: GameState):
        """
        Mark the end of the game and update rewards for all transitions.

        Args:
            state: GameState object.
        """
        # Calculate terminal reward based on the final state
        terminal_reward = calculate_rewards_state(state, immediate=False)
        if DEBUG:
            print(f"DEBUG: Terminal reward calculated: {terminal_reward}")

        for i, (s, a, _, next_s, _) in enumerate(self.experience_buffer):
            done = (i == len(self.experience_buffer) - 1)  # Only the last transition is marked as done
            self.experience_buffer[i] = (s, a, terminal_reward if done else 0, next_s, done)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if log:
            print(f"Epsilon decayed to: {self.epsilon}")
            print(f"Final transition: Done={done}, Reward={terminal_reward}")

    def reset(self):
        """
        Reset the agent state for a new game.
        """
        self.previous_state = None
        self.previous_action = None
        # self.experience_buffer.clear()
        if DEBUG:
            print("DEBUG: Agent state reset for a new game.")
