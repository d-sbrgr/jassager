# rl_agent.py

import torch

from jass.game.game_util import full_to_trump, convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.const import *

from bots.mcts_bots.util.mcts_implementation import ISMCTS
from bots.rl_bots.util.encode_game_obs import *
from bots.rl_bots.util.reward_system import calculate_rewards

class RLAgent(Agent):
    def __init__(self, model):
        self.model = model
        self.experience_buffer = []
        self.previous_state = None
        self.previous_action = None
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Decide on trump using the neural network.
        :param obs: The current game observation.
        :return: Integer representing the trump action.
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

    def action_play_card(self, obs: GameObservation):
        """
        Decide which card to play.
        :param obs: The current game observation.
        :return: The selected card to play.
        """
        current_state = encode_game_observation(obs)

        # Choose action
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs)
        )
        if len(valid_moves) == 1:
            action = valid_moves[0]
        else:
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy, _ = self.model(state_tensor)
            action_probabilities = policy.numpy().squeeze()

            masked_probabilities = np.zeros_like(action_probabilities)
            masked_probabilities[valid_moves] = action_probabilities[valid_moves]
            action = int(np.argmax(masked_probabilities))

        # Store transition for the previous move
        if self.previous_state is not None and self.previous_action is not None:
            reward = calculate_rewards(obs, immediate=True)  # Use the current observation to calculate reward
            done = obs.nr_played_cards == 36  # Game ends when all 36 cards are played
            self.store_transition(self.previous_state, self.previous_action, reward, current_state, done)

        # Update previous state and action
        self.previous_state = current_state
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
        :param obs: The final game observation to calculate the terminal reward.
        """
        terminal_reward = calculate_rewards(obs, immediate=False)
        for i, (state, action, _, next_state, _) in enumerate(self.experience_buffer):
            done = i == len(self.experience_buffer) - 1  # Mark the last transition as done
            self.experience_buffer[i] = (state, action, terminal_reward if done else 0, next_state, done)