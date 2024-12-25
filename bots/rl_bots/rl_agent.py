import numpy as np
import torch

from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from bots.rl_bots.util.encode_game_obs import encode_game_observation

# Debug flag
debug = False

# Log flag
log = False

if debug:
    print(" ===== RLAgent.py in debug mode =====")

class RLAgent(Agent):
    def __init__(self, model, epsilon=0, epsilon_decay=0, min_epsilon=0):
        """
        Initialize the RLAgent for testing/operation.

        Args:
            model: Trained neural network model for policy and value predictions.
            epsilon: Initial exploration rate for epsilon-greedy policy.
            epsilon_decay: Decay factor for epsilon.
            min_epsilon: Minimum value for epsilon.
        """
        self.model = model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Decide the trump suit based on the trained model.

        Args:
            obs (GameObservation): The current game observation.

        Returns:
            int: The selected trump suit.
        """
        if debug:
            print(f"Debug: Calling action_trump, Trump: {obs.trump}")

        encoded_state = encode_game_observation(obs)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

        # Predict trump probabilities using the model
        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        # Valid trump actions
        valid_trumps = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]

        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            trump_action = np.random.choice(valid_trumps)
            if log:
                print(f"Exploring: Random trump action selected: {trump_action}")
        else:
            trump_action = max(valid_trumps, key=lambda t: policy[0, t].item() if 0 <= t < 6 else float('-inf'))

            if log:
                print(f"Exploiting: Selected trump action: {trump_action}")

        valid_trumps = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]
        assert trump_action in valid_trumps, f"Invalid trump action {trump_action}, valid trumps: {valid_trumps}"
        assert 0 <= trump_action < 6, f"Trump action {trump_action} is out of bounds"

        return trump_action

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Decide which card to play based on the trained model.

        Args:
            obs (GameObservation): The current game observation.

        Returns:
            int: The selected card to play.
        """
        if debug:
            print(f"Debug: Calling action_play_card, Cards played: {obs.nr_played_cards}")

        encoded_state = encode_game_observation(obs)
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs)
        )

        if len(valid_moves) == 1:
            action = valid_moves[0]
            if log:
                print(f"Single valid move available: {action}")
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
                if log:
                    print(f"Exploiting: Selected action: {action} with probabilities {action_probabilities}")

        assert action in valid_moves, f"Invalid card action {action}, valid moves: {valid_moves}"
        assert 0 <= action < 36, f"Card action {action} is out of bounds"

        return action

    def reset(self):
        """
        Reset the agent state for a new game.
        """
        if debug:
            print("Debug: Agent state reset for a new game.")
