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


class MCTSGameSim(GameSim):

    def __init__(self, rule: GameRule, agent: FullHeuristicTableView, team: int):
        super().__init__(rule)
        self._agent = agent
        self._team: int = team
        self.rule.get_valid_actions_from_state()

    def run_simulation(self):
        while not self.is_done():
            self.action_play_card(self._agent.action_play_card(self.get_observation()))

    def get_score(self) -> float:
        return (self.state.points[self._team]) / 157


class MCTSGameState(GameState):

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
        self.children = {}  # action -> Node
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, game_state: MCTSGameState):
        legal_actions =
        return all(legal_action in self.children.keys() for legal_action in legal_actions)

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        return max(self.children.values(), key=lambda node: node.visits)


class ISMCTS:
    def __init__(self, obs: GameObservation, iterations: int = None, time: float = None):
        if time is None and iterations is None:
            raise RuntimeError("ISMCTS must be terminated either after elapsed time or number of iterations")
        self.iterations = iterations
        self.obs = obs
        self.root = ISMCTSNode()
        self.sim = MCTSGameSim(RuleSchieber(), FullHeuristicTableView(), 0 if obs.player_view in (NORTH, SOUTH) else 1)
        self.possible_cards = MCTSGameState.get_possible_cards(obs)
    def search(self):
        for _ in range(self.iterations):
            self.sim.init_from_state(MCTSGameState.from_observation(self.obs, self.possible_cards))
            node, state = self.selection(self.root, self.sim)
            reward = self.simulation(state)
            self.backpropagate(node, reward)
        return self.root.most_visited_child().action

    def selection(self, node, sim: MCTSGameSim):
        while not sim.is_done():
            if node.is_fully_expanded(state):
                node = node.best_child()
                state = state.perform_action(node.action)
            else:
                return self.expand(node, state)
        return node, state

    def expand(self, node, state):
        tried_actions = node.children.keys()
        possible_actions = state.get_legal_actions()
        for action in possible_actions:
            if action not in tried_actions:
                new_state = state.perform_action(action)
                information_set = new_state.get_information_set(self.player)
                child_node = ISMCTSNode(information_set, parent=node, action=action)
                node.children[action] = child_node
                return child_node, new_state
        # Should not reach here
        return node, state

    def simulation(self, state):
        # Sample a hidden information state consistent with the information set
        sampled_state = self.sample_state(state)
        while not sampled_state.is_terminal():
            action = random.choice(sampled_state.get_legal_actions())
            sampled_state = sampled_state.perform_action(action)
        return sampled_state.get_reward(self.player)

    def sample_state(self, state):
        # Implement state sampling consistent with the information set
        # This is game-specific and needs to be implemented accordingly
        return state  # Placeholder

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
