from __future__ import annotations

import math
import random

from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_sim import GameSim
from jass.game.game_state import GameState
from jass.game.game_state_util import state_from_observation, calculate_points_from_tricks
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list


class MCTSGameState:
    def __init__(self, state: GameState, sim: GameSim, team: int) -> None:
        self._simulation = sim
        self._state = state
        self._team = team
        self._simulation.init_from_state(state)

    def get_reward(self) -> float:
        return float(self._state.points[self._team] / 157)

    def get_legal_actions(self) -> list[int]:
        return convert_one_hot_encoded_cards_to_int_encoded_list(
            self._simulation.rule.get_valid_cards_from_state(self._state))

    @property
    def is_terminal(self) -> bool:
        return self._state.nr_played_cards == 36

    def perform_action(self, action) -> MCTSGameState:
        self._simulation.action(action)
        return MCTSGameState(self._simulation.state, self._simulation, self._team)


class ISMCTSNode:
    def __init__(self, parent: ISMCTSNode | None = None, action: int | None = None):
        self.parent = parent
        self.action = action
        self.children: dict[int, ISMCTSNode] = {}
        self.visits = 0
        self.available = 0
        self.reward = 0.0
        self._state = None

    def is_fully_expanded(self, state: MCTSGameState) -> bool:
        self._state = state
        return all(legal_action in self.children.keys() for legal_action in state.get_legal_actions())

    def best_child(self, c_param=1.4) -> ISMCTSNode:
        children = list(self.children.values())
        return max(
            children,
            key=lambda child: child.get_ucb1_score()
        )

    def get_ucb1_score(self, c_param=1.4) -> float:
        return self.reward / self.visits + c_param * math.sqrt(math.log(self.available) / self.visits)

    def most_visited_child(self):
        return max(self.children.values(), key=lambda node: node.visits)

    def create_random_child(self) -> ISMCTSNode:
        action = random.choice(self._state.get_legal_actions())
        child = ISMCTSNode(self, action)
        self.children[action] = child
        return child

    def get_available_siblings(self) -> list[ISMCTSNode]:
        return [c for c in self.children.values() if c.action in self._state.get_legal_actions()]

    @staticmethod
    def get_random_determinization(obs: GameObservation, sim: GameSim) -> MCTSGameState:
        hands = np.zeros([4, 36], np.int32)
        hands[obs.player_view, :] = obs.hand
        possible_cards = np.ones([4, 36], int)
        # Remove all played cards and if applicable, color types from player's possible hands
        for index, card in enumerate(obs.tricks.flatten()):
            if card == -1:  # No more cards played
                break

            possible_cards[:, card] = 0

            if not index % 4:  # First card in trick
                player = obs.trick_first_player[index // 4]
                first_color = color_of_card[card]
                continue

            player = next_player[player]
            color = color_of_card[card]

            if color != first_color:
                offset = color_offset[first_color]
                if first_color == obs.trump and possible_cards[player, offset + J_offset] == 1:
                    possible_cards[player, offset: offset + 9] = 0
                    possible_cards[player, offset + J_offset] = 1
                else:
                    possible_cards[player, offset: offset + 9] = 0

        # Remove all cards in player_view's hand
        possible_cards[:, convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)] = 0
        possible_cards[obs.player_view, :] = 0  # Player_view's cards are already determined
        remaining_cards = convert_one_hot_encoded_cards_to_int_encoded_list(np.bitwise_or.reduce(possible_cards))

        # How many cards each player must hold
        card_count = [8 - obs.nr_tricks] * 4
        player = obs.player_view
        for i in range(4 - obs.nr_played_cards):
            card_count[player] += 1
            player = next_player[player]
        card_count[obs.player_view] = 0

        while True:
            random.shuffle(remaining_cards)
            player_cards = [
                remaining_cards[:card_count[0]],
                remaining_cards[card_count[0]:card_count[0] + card_count[1]],
                remaining_cards[card_count[0] + card_count[1]:card_count[0] + card_count[1] + card_count[2]],
                remaining_cards[card_count[0] + card_count[1] + card_count[2]:],
            ]
            if all(
                    [possible_cards[player, cards].all() == 1 for player, cards in enumerate(player_cards)]
            ):
                for player, cards in enumerate(player_cards):
                    hands[player, cards] = 1
                break

        return MCTSGameState(state_from_observation(obs, hands), sim, 0 if obs.player_view in (NORTH, SOUTH) else 1)


class ISMCTS:
    def __init__(self, obs: GameObservation, rule: GameRule, iterations: int = 1000):
        self.iterations = iterations
        self.obs = obs
        self.sim = GameSim(rule)
        self.root = ISMCTSNode()

    def search(self):
        for _ in range(self.iterations):
            state = self.root.get_random_determinization(self.obs, self.sim)
            node, state = self.selection(self.root, state)
            if not node.is_fully_expanded(state):
                node, state = self.expand(node, state)
            reward = self.simulation(state)
            self.backpropagate(node, reward)
        return self.root.most_visited_child().action

    @staticmethod
    def selection(node: ISMCTSNode, state: MCTSGameState) -> tuple[ISMCTSNode, MCTSGameState]:
        while not state.is_terminal and node.is_fully_expanded(state):
            node = node.best_child()
            state = state.perform_action(node.action)
        return node, state

    @staticmethod
    def expand(node, state) -> tuple[ISMCTSNode, MCTSGameState]:
        node = node.create_random_child()
        state = state.perform_action(node.action)
        return node, state

    @staticmethod
    def simulation(state: MCTSGameState):
        while not state.is_terminal:
            action = random.choice(state.get_legal_actions())
            state = state.perform_action(action)
        return state.get_reward()

    @staticmethod
    def backpropagate(node: ISMCTSNode, reward: float):
        while node.parent is not None:
            node.visits += 1
            node.available += 1
            node.reward += reward
            for sibling in node.get_available_siblings():
                sibling.available += 1
            node = node.parent
