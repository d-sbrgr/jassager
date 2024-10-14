from __future__ import annotations

import numpy as np

import math
import time
from collections.abc import Iterable

from jass.game.game_observation import GameObservation

from bots.mcts_bots.util.mcts_game_state import MCTSGameState

DEBUG = True


class ISMCTS:
    """
    Implements the Information Set Monte Carlo Tree Search (ISMCTS) algorithm for imperfect information games.

    This class orchestrates the MCTS process, handling the tree search iterations,
    and ultimately selecting the best action to take from the current game observation.
    """
    def __init__(self, obs: GameObservation, max_time: float = 2.0):
        """
        Initialize the ISMCTS algorithm.

        :param obs: The current game observation available to the player.
        :param max_time: The time to elapse after which a result shall be returned.
        """
        np.random.seed(1)
        self.start = time.time()
        self.max_time = max_time
        self.obs = obs                  # Current game observation
        self.root = ISMCTSNode()        # Root node of the search tree
        self._current_node = None

    def search(self):
        """
        Execute the ISMCTS algorithm to find the best action from the current observation.

        :return: The action (card index) corresponding to the most promising move.
        """
        det = []
        alg = []
        while time.time() - self.start < self.max_time:
            t_strt = time.perf_counter()

            self._current_node = self.root  # Set root node as current node
            self.root.state = MCTSGameState.random_state_from_obs(self.obs)  # Step 1: Determinization - sample a possible complete game state

            t_det = time.perf_counter()

            self.selection()  # Step 2: Selection - traverse the tree to find a node to expand
            self.expand()  # Step 3: Expansion - add a new child node if the node is not fully expanded
            self.simulation()  # Step 4: Simulation - simulate a random playout from the node
            self.backpropagate()  # Step 5: Backpropagation - update the nodes with the simulation result

            t_stop = time.perf_counter()
            det.append(t_det - t_strt)
            alg.append(t_stop - t_det)

        if DEBUG:
            self._benchmark_print(det, alg)

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
        self._current_node.state.run_internal_simulation()

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

    def _benchmark_print(self, det, alg):
        print(
            f"{sum(det) / len(det):.6f} | {max(det):.6f} | {min(det):.6f}"
            f" ||--|| "
            f"{sum(alg) / len(alg):.6f} | {max(alg):.6f} | {min(alg):.6f} "
            f" ||--|| "
            f"{self.obs.nr_played_cards}"
        )


class ISMCTSNode:
    """
    Represents a node in the Information Set Monte Carlo Tree Search (ISMCTS) tree.

    Each node corresponds to a game state resulting from an action and contains statistics
    used to guide the tree search, such as visit counts and accumulated rewards.
    """
    def __init__(self, parent: ISMCTSNode | None = None, action: int | None = None):
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
        self.state: MCTSGameState | None = None          # Associated game state

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
        return self.reward / self.visits + 1.3 * math.sqrt(math.log(self.available) / self.visits)

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
        """
        # Get legal actions from the current state
        child = ISMCTSNode(self, np.random.choice(self.state.legal_actions))
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
