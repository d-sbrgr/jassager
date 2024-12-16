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
            print(f"Exploring: Random trump action selected: {trump_action}")
        else:
            trump_action = max(valid_trumps, key=lambda t: policy[0, t].item())
            print(f"Exploiting: Selected trump action: {trump_action}")

        return trump_action

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Decide which card to play based on the trained model.

        Args:
            obs (GameObservation): The current game observation.

        Returns:
            int: The selected card to play.
        """
        print(f"Debug: Calling action_play_card, Cards played: {obs.nr_played_cards}")

        encoded_state = encode_game_observation(obs)
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs)
        )

        if len(valid_moves) == 1:
            action = valid_moves[0]
            print(f"Single valid move available: {action}")
        else:
            state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

            if np.random.rand() < self.epsilon:
                action = np.random.choice(valid_moves)
                print(f"Exploring: Random action selected: {action}")
            else:
                with torch.no_grad():
                    policy_logits, _ = self.model(state_tensor)
                masked_logits = policy_logits.clone()
                mask = torch.zeros_like(masked_logits)
                mask[0, valid_moves] = 1
                masked_logits[mask == 0] = float('-inf')
                action_probabilities = torch.softmax(masked_logits, dim=-1).squeeze()
                action = torch.argmax(action_probabilities).item()
                print(f"Exploiting: Selected action: {action} with probabilities {action_probabilities}")

        return action

    def reset(self):
        """
        Reset the agent state for a new game.
        """
        print("Debug: Agent state reset for a new game.")
