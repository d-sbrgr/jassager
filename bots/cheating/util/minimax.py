from __future__ import annotations

import math

from jass.game.rule_schieber import RuleSchieber
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.const import *


class Minimax:
    _sim = GameSim(RuleSchieber())
    def __init__(self, state: GameState, depth: int = 10):
        self._team = 0 if state.player in (NORTH, SOUTH) else 1
        self._max_depth = depth
        self._root = MiniMaxNode(True)
        self._build_tree(state)

    def is_maximizing(self, player: int):
        if self._team == 1:
            return player in (NORTH, SOUTH)
        return player in (EAST, WEST)

    def _build_tree(self, state: GameState):
        self._expand_for_node(self._root, state, 0)

    def _expand_for_node(self, node: MiniMaxNode, state: GameState, depth: int):
        if depth > self._max_depth:
            # Do heuristics based simulation
            node.value = state.points[self._team]
        else:
            for action in convert_one_hot_encoded_cards_to_int_encoded_list(self._sim.rule.get_valid_actions_from_state(state)):
                self._sim.init_from_state(state)
                self._sim.action(action)
                if self._sim.state.nr_played_cards == 36:
                    node.add_child(action, MiniMaxNode(True, self._sim.state.points[self._team]))
                else:
                    child = MiniMaxNode(self.is_maximizing(self._sim.state.player), None)
                    node.add_child(action, child)
                    self._expand_for_node(child, self._sim.state, self._sim.state.nr_tricks)

    def get_best_action(self) -> int:
        value = self._root.maximize(-1, 158)
        return self._root.get_action_for_child_value(value)


class MiniMaxNode:
    def __init__(self, is_maximizing: bool, value: float = None):
        self.value = value
        self._maximizing = is_maximizing
        self._children: dict[int, MiniMaxNode] = {}

    def get_action_for_child_value(self, value: float) -> int:
        for action, child in self._children.items():
            if child.value == value:
                return action

    def add_child(self, action: int, child: MiniMaxNode):
        self._children[action] = child

    @property
    def is_maximizing(self) -> bool:
        return self._maximizing

    @property
    def is_leaf(self) -> bool:
        return not bool(self._children)

    def maximize(self, alpha: float, beta: float) -> float:
        if not self.is_leaf:
            best = -1
            for child in self._children.values():
                if not child.is_maximizing:
                    best = max(child.minimize(alpha, beta), best)
                else:
                    best = max(child.maximize(alpha, beta), best)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            self.value = best
        return self.value

    def minimize(self, alpha: float, beta: float) -> float:
        if not self.is_leaf:
            best = 158
            for child in self._children.values():
                if child.is_maximizing:
                    best = min(child.maximize(alpha, beta), best)
                else:
                    best = min(child.minimize(alpha, beta), best)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            self.value = best
        return self.value
