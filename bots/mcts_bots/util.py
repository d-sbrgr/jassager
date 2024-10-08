from __future__ import annotations

import numpy as np

import math
import random

from jass.game.game_rule import GameRule
from jass.game.game_state_util import state_from_observation
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.game.const import *
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from ..heuristic_bots.full_heuristic_v2 import FullHeuristicTableView


class MCTSGameState(GameState):
    def get_reward(self) -> float:
        pass

    def get_legal_actions(self) -> tuple[int, ...]:
        pass

    def perform_action(self, action) -> MCTSGameState:
        pass

    @property
    def is_terminal(self) -> bool:
        pass

    @classmethod
    def from_observation(cls, obs: GameObservation, possible_cards: np.ndarray) -> GameState:
        unplayed_cards = np.ones(36, np.int32)
        for card in obs.tricks.flatten():
            if card == -1:
                break
            unplayed_cards[card] = 0

        hands = np.zeros([4, 36], np.int32)
        hands[obs.player, :] = obs.hand

        cards = convert_one_hot_encoded_cards_to_int_encoded_list(unplayed_cards)
        while not cls.check_cards_against_possible(hands, possible_cards):
            random.shuffle(cards)
            player = next_player[obs.player]
            for card in cards:
                if not hands[obs.player_view, card]:
                    hands[player, card] = 1
                    player = next_player[player]
                    if player == obs.player_view:
                        player = next_player[player]

        return state_from_observation(obs, hands)

    @classmethod
    def check_cards_against_possible(cls, cards: np.ndarray, possible_cards: np.ndarray) -> bool:
        for i in range(4):
            if np.bitwise_and(cards[i], possible_cards[i]) != cards[i]:
                return False
        return True

    @classmethod
    def get_possible_cards(cls, obs: GameObservation) -> np.ndarray:
        cards = np.ones([4, 36], np.int32)
        trump = obs.trump
        for trick, trick_cards in enumerate(obs.tricks):
            player = obs.trick_first_player[trick]
            first_color = color_of_card[trick_cards[0]]
            for card in trick_cards[1:]:
                if card == -1:
                    break
                cards[:,card] = 0
                player = next_player[player]
                color = color_of_card[card]
                if color != first_color:
                    offset = color_offset[first_color]
                    if color != trump:
                        cards[player, offset:offset + 9] = 0
                    elif first_color == trump:
                        cards[player, offset:offset + 9] = 0
                        cards[player, offset + J_offset] = 1
        return cards


class ISMCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children: dict[int, ISMCTSNode] = {}
        self.visits = 0
        self.available = 0
        self.reward = 0.0

    def is_fully_expanded(self, state: MCTSGameState):
        legal_actions = state.get_legal_actions()
        return all(legal_action in self.children.keys() for legal_action in legal_actions)

    def best_child(self, c_param=1.4):
        # TODO: Cleanly implement this
        choices_weights = [
            (child.reward / child.visits) + c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children.values(), key=lambda node: node.visits)

    def get_random_determinization(self) -> MCTSGameState:
        pass

    def get_random_action(self):
        pass

    def get_available_siblings(self) -> list[ISMCTSNode]:
        pass


class ISMCTS:
    def __init__(self, obs: GameObservation, rule: GameRule, iterations: int = 1000):
        self.iterations = iterations
        self.obs = obs
        self.root = ISMCTSNode.from_observation(obs, rule)
        # self.possible_cards = MCTSGameState.get_possible_cards(obs)

    def search(self):
        for _ in range(self.iterations):
            state = self.root.get_random_determinization()
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
            state = state.perfom_action(node.action)
        return node, state

    @staticmethod
    def expand(node, state) -> tuple[ISMCTSNode, MCTSGameState]:
        action = node.get_random_action(state)
        child = ISMCTSNode(node, action)
        node.children[action] = child
        state = state.perform_action(action)
        return node, state

    @staticmethod
    def simulation(state: MCTSGameState):
        while not state.is_terminal:
            action = random.choice(state.get_legal_actions())
            state = state.perform_action(action)
        return state.get_reward()

    @staticmethod
    def backpropagate(node: ISMCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            node.reward += reward
            for sibling in node.get_available_siblings():
                sibling.available += 1
            node = node.parent
