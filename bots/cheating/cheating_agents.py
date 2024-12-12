import numpy as np
import torch

from jass.agents.agent_cheating import AgentCheating
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from bots.rl_bots.util.reward_system import calculate_rewards_state
from bots.rl_bots.util.encode_game_state import encode_game_state


class HeuristicAgentCheating(AgentCheating):
    def __init__(self):
        super().__init__()
        from bots.heuristic_bots.full_heuristic_v2 import FullHeuristicTableView
        self.bot = FullHeuristicTableView()

    def action_trump(self, state: GameState):
        return self.bot.action_trump(state)

    def action_play_card(self, state: GameState):
        return self.bot.action_play_card(state)


class RandomAgentCheating(AgentCheating):
    def __init__(self):
        super().__init__()
        from bots.random_bot.full_random import RandomAgent
        self.bot = RandomAgent()

    def action_trump(self, state: GameState):
        return self.bot.action_trump(state)

    def action_play_card(self, state: GameState):
        return self.bot.action_play_card(state)


class RLAgentCheating(AgentCheating):
    def __init__(self, model, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.experience_buffer = []
        self.previous_state = None
        self.previous_action = None
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        encoded_state = encode_game_state(state)
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        trump_action = torch.argmax(policy).item()
        if trump_action >= 6:
            trump_action = DIAMONDS


        return trump_action

    def action_play_card(self, state: GameState) -> int:
        current_hand = state.hands[state.player]

        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards(current_hand, state.current_trick, state.nr_cards_in_trick, state.trump)
        )

        if len(valid_moves) == 1 or np.random.rand() < self.epsilon:
            action = np.random.choice(valid_moves)

        else:
            encoded_state = encode_game_state(state)
            state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy, _ = self.model(state_tensor)
            action_probabilities = policy.numpy().squeeze()
            masked_probabilities = np.zeros_like(action_probabilities)
            masked_probabilities[valid_moves] = action_probabilities[valid_moves]
            action = int(np.argmax(masked_probabilities))

        reward = calculate_rewards_state(state, immediate=True)
        done = state.nr_played_cards == 36
        if self.previous_state is not None and self.previous_action is not None:
            self.store_transition(self.previous_state, self.previous_action, reward, encode_game_state(state), done)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.previous_state = encode_game_state(state)
        self.previous_action = action

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))
