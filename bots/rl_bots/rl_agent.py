import torch
import numpy as np
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.const import *

from bots.rl_bots.util.encode_game_obs import encode_game_observation
from bots.rl_bots.util.reward_system import calculate_rewards_obs

class RLAgent(Agent):
    def __init__(self, model, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        """
        Initialize the RLAgent.

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

    def action_trump(self, obs: GameObservation) -> int:
        """
        Decide on trump using the neural network.

        Args:
            obs: GameObservation object representing the current observation.

        Returns:
            Integer representing the trump action.
        """
        encoded_state = encode_game_observation(obs)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

        # Predict trump probabilities using the model
        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        # Choose the trump with the highest probability
        trump_action = torch.argmax(policy).item()

        # Ensure the action corresponds to a valid trump
        if trump_action >= 6:  # Adjust if your trumps include UNE_UFE and OBE_ABE
            trump_action = DIAMONDS  # Fallback to Diamonds if invalid

        return trump_action

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Decide on a card to play based on the current GameObservation.

        Args:
            obs: GameObservation object.

        Returns:
            Integer representing the selected action.
        """
        encoded_state = encode_game_observation(obs)
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs)
        )

        if len(valid_moves) == 1 or np.random.rand() < self.epsilon:
            # Explore: Pick a random valid move
            action = np.random.choice(valid_moves)
        else:
            # Exploit: Use the model to pick the best action
            state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy, _ = self.model(state_tensor)
            action_probabilities = policy.numpy().squeeze()
            masked_probabilities = np.zeros_like(action_probabilities)
            masked_probabilities[valid_moves] = action_probabilities[valid_moves]
            action = int(np.argmax(masked_probabilities))

        # Calculate reward and store transition
        reward = calculate_rewards_obs(obs, immediate=True)
        done = obs.nr_played_cards == 36  # Done if all cards are played
        if self.previous_state is not None and self.previous_action is not None:
            self.store_transition(self.previous_state, self.previous_action, reward, encoded_state, done)

        # Update epsilon during decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update previous state and action
        self.previous_state = encoded_state
        self.previous_action = action

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the experience buffer.
        """
        self.experience_buffer.append((state, action, reward, next_state, done))

    def finalize_game(self, obs: GameObservation):
        """
        Mark the end of the game and update rewards for all transitions.

        Args:
            obs: GameObservation object.
        """
        terminal_reward = calculate_rewards_obs(obs, immediate=False)
        for i, (state, action, _, next_state, _) in enumerate(self.experience_buffer):
            done = i == len(self.experience_buffer) - 1  # Mark the last transition as done
            self.experience_buffer[i] = (state, action, terminal_reward if done else 0, next_state, done)
