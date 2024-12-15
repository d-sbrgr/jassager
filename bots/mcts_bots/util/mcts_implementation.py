from __future__ import annotations

import numpy as np
import torch.nn as nn
from typing import Callable, Type

import math
import time
from collections.abc import Iterable

from jass.game.game_observation import GameObservation

from bots.mcts_bots.util.mcts_game_state import PureMCTSGameState, MCTSGameState

MAX_SEARCH_DURATION = 9.5


class ISMCTS:
    """
    Implements the Information Set Monte Carlo Tree Search (ISMCTS) algorithm for imperfect information games.

    This class orchestrates the MCTS process, handling the tree search iterations,
    and ultimately selecting the best action to take from the current game observation.
    """
    def __init__(self, obs: GameObservation, state: Type[MCTSGameState], c_param: float = 1.3, model: nn.Module = None, conversion: Callable = None):
        """
        Initialize the ISMCTS algorithm.

        :param obs: The current game observation available to the player.
        :param iterations: Number of iterations the bot should run.
                           If not provided, the number of iterations are calculated
        """
        np.random.seed(1)
        self.start = time.time()
        self.obs = obs                  # Current game observation
        self.iterations = self._get_number_of_iterations()
        self.root = ISMCTSNode(c_param=c_param)        # Root node of the search tree
        self._model = model
        self._state_factory = state
        self._conversion = conversion
        self._current_node: ISMCTSNode = None

    def search(self):
        """
        Execute the ISMCTS algorithm to find the best action from the current observation.

        :return: The action (card index) corresponding to the most promising move.
        """
        det = []
        alg = []
        for _ in range(self.iterations):
            self._current_node = self.root  # Set root node as current node
            self.root.state = self._state_factory.random_state_from_obs(self.obs, self._model)  # Step 1: Determinization - sample a possible complete game state
            self.selection()  # Step 2: Selection - traverse the tree to find a node to expand
            self.expand()  # Step 3: Expansion - add a new child node if the node is not fully expanded
            self.simulation()  # Step 4: Simulation - simulate a random playout from the node
            self.backpropagate()  # Step 5: Backpropagation - update the nodes with the simulation result

            if time.time() - self.start > MAX_SEARCH_DURATION:
                break

        # After all iterations, select the action with the most visits at the root
        return self.root.get_most_visited_child().action

    def selection(self):
        """
        Traverse the tree, selecting child nodes based on UCB1 until a leaf node is reached.
        """
        while self._current_node.is_fully_expanded and not self._current_node.is_terminal:
            self._current_node = self._current_node.get_best_child()

    def expand(self):
        """
        Expand the current node by adding a new child node for an unexplored action.
        """
        if self._current_node.is_terminal:
            return
        self._current_node = self._current_node.get_random_new_child()

    def simulation(self):
        """
        Simulate a random playout from the current state to a terminal state.
        """
        self._current_node.state.run_internal_simulation(self._conversion)

    def backpropagate(self):
        """
        Backpropagate the simulation results up the tree, updating node statistics.
        """
        reward = self._current_node.state.get_reward()
        while self._current_node.parent is not None:
            self._current_node.visits += 1
            self._current_node.available += 1
            self._current_node.reward += reward
            for sibling in self._current_node.get_available_siblings():
                sibling.available += 1
            self._current_node = self._current_node.parent

    def _get_number_of_iterations(self) -> int:
        return int(4000 + (100000 / 36 * (36 - self.obs.nr_played_cards)))


class ISMCTSNode:
    """
    Represents a node in the Information Set Monte Carlo Tree Search (ISMCTS) tree.

    Each node corresponds to a game state resulting from an action and contains statistics
    used to guide the tree search, such as visit counts and accumulated rewards.
    """
    def __init__(self, parent: ISMCTSNode | None = None, action: int | None = None, c_param: float = 1.3):
        """
        Initialize an ISMCTS node.

        :param parent: The parent node (None if this is the root node).
        :param action: The action taken to reach this node from the parent.
        """
        self.parent = parent        # Parent node in the tree
        self.action = action        # Action that led to this node
        self.children: dict[int, ISMCTSNode] = {}       # Child nodes mapped by action
        self.visits = 0             # Number of times this node was visited
        self.available = 0          # Number of times this node was available for selection
        self.reward = 0.0           # Total reward accumulated through this node
        self.state: MCTSGameState = None          # Associated game state
        self.c_param = c_param

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions from this node have been expanded.

        :return: True if all legal actions have corresponding child nodes, False otherwise.
        """
        return all(action in self.children for action in self.state.legal_actions)

    @property
    def ucb1_score(self) -> float:
        """
        Calculate the UCB1 score for this node.

        :return: The UCB1 score as a float.
        """
        return self.reward / self.visits + self.c_param * math.sqrt(math.log(self.available) / self.visits)

    def get_most_visited_child(self):
        """
        Get the child node with the highest visit count.

        This is typically used to select the final action after the search is complete.

        :return: The child node with the most visits.
        """
        return max(self.children.values(), key=lambda node: node.visits)

    def get_best_child(self) -> ISMCTSNode:
        """
        Select the best child node based on the Upper Confidence Bound (UCB1) score.

        :return: The child node with the highest UCB1 score.
        """
        child = max(
            self.children.values(),
            key=lambda c: c.ucb1_score
        )
        child.state = self.state.perform_action(child.action)
        return child

    def get_random_new_child(self) -> ISMCTSNode:
        """
        Expand the node by creating a new child node with a random unexplored action.

        :return: The newly created child node.

        Raises:
            ValueError: If there are no unexplored actions available to create a new child node.

        """
        random_action = np.random.choice(
            [action for action in self.state.legal_actions if action not in self.children]
        )
        child = ISMCTSNode(self, random_action, c_param=self.c_param)
        child.state = self.state.perform_action(child.action)
        self.children[child.action] = child
        return child

    def get_available_siblings(self) -> Iterable[ISMCTSNode]:
        """
        Retrieve a list of sibling nodes that are available for selection.

        :return: A list of sibling ISMCTSNode instances.
        """
        siblings = set(self.parent.children.values()).difference({self})
        return (s for s in siblings if s.action in self.parent.state.legal_actions)
