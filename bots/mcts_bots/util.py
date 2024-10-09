from __future__ import annotations

import copy
import math
import random

import numpy as np
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

    def get_reward(self) -> float:
        return float(self._state.points[self._team] / 157)

    def get_legal_actions(self) -> list[int]:
        return convert_one_hot_encoded_cards_to_int_encoded_list(
            self._simulation.rule.get_valid_cards_from_state(self._state))

    @property
    def is_terminal(self) -> bool:
        return self._state.nr_played_cards == 36

    def perform_action(self, action) -> MCTSGameState:
        self._simulation.init_from_state(self._state)
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
        return [c for c in self.parent.children.values() if c.action in self.parent._state.get_legal_actions() and c.action != self.action]

    def set_state(self, state: MCTSGameState):
        self._state = state

    @staticmethod
    def get_random_determinization(obs: GameObservation, sim: GameSim) -> MCTSGameState:
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
        possible_cards[obs.player, :] = 0  # Player_view's cards are already determined
        remaining_cards = convert_one_hot_encoded_cards_to_int_encoded_list(np.bitwise_or.reduce(possible_cards))

        # How many cards each player must hold
        card_count = [8 - obs.nr_tricks] * 4
        player = obs.player
        for i in range(4 - obs.nr_cards_in_trick):
            card_count[player] += 1
            player = next_player[player]
        card_count[obs.player] = 0

        hands = np.zeros([4, 36], np.int32)
        hands[obs.player, :] = obs.hand

        c = 0
        remaining_cards_copy = list(remaining_cards)
        card_count_copy = list(card_count)
        while len(remaining_cards) > 0:
            c += 1
            if c > 100:
                raise ValueError()
            card_count = list(card_count_copy)
            remaining_cards = list(remaining_cards_copy)
            for i in range(1, 4):
                cards, remaining_cards = remaining_cards, []
                for card in cards:
                    possible_players = possible_cards[:, card]
                    if possible_players.sum() == i:
                        players, = np.where(possible_players == 1)
                        players = players.tolist()
                        random.shuffle(players)
                        for player in players:
                            if card_count[player] > 0:
                                hands[player, card] = 1
                                card_count[player] -= 1
                                break
                        else:
                            remaining_cards.append(card)
                    else:
                        remaining_cards.append(card)

        if obs.nr_played_cards + hands.sum() != 36:
            raise ValueError()

        return MCTSGameState(state_from_observation(obs, hands), sim, 0 if obs.player_view in (NORTH, SOUTH) else 1)


class ISMCTS:
    def __init__(self, obs: GameObservation, rule: GameRule, iterations: int = 1000):
        random.seed(1)
        self.iterations = iterations
        self.obs = obs
        self.sim = GameSim(rule)
        self.root = ISMCTSNode()

    def search(self):
        for _ in range(self.iterations):
            state = self.root.get_random_determinization(self.obs, self.sim)
            self.root.set_state(state)
            node, state = self.selection(self.root, state)
            if not node.is_fully_expanded(state) and not state.is_terminal:
                node, state = self.expand(node, state)
            reward = self.simulation(state)
            self.backpropagate(node, reward)
        return self.root.most_visited_child().action

    @staticmethod
    def selection(node: ISMCTSNode, state: MCTSGameState) -> tuple[ISMCTSNode, MCTSGameState]:
        while not state.is_terminal and node.is_fully_expanded(state):
            node = node.best_child()
            state = state.perform_action(node.action)
            node.set_state(state)
        return node, state

    @staticmethod
    def expand(node, state) -> tuple[ISMCTSNode, MCTSGameState]:
        child_node = node.create_random_child()
        child_state = state.perform_action(child_node.action)
        child_node.set_state(child_state)
        return child_node, child_state

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
