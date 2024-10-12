from __future__ import annotations

import math
import random
import time

from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule

from .mcts_game_state import MCTSGameState, mcts_state_from_observation
from .util import get_random_determinization


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
        self._state = None          # Associated game state

    def is_fully_expanded(self, state: MCTSGameState) -> bool:
        """
        Check if all possible actions from this node have been expanded.

        :param state: The current game state.
        :return: True if all legal actions have corresponding child nodes, False otherwise.
        """
        return all(legal_action in self.children for legal_action in state.get_legal_actions())

    def best_child(self, c_param=1.4) -> ISMCTSNode:
        """
        Select the best child node based on the Upper Confidence Bound (UCB1) score.

        :param c_param: Exploration parameter balancing exploitation and exploration.
        :return: The child node with the highest UCB1 score.
        """
        return max(
            self.children.values(),
            key=lambda child: child.get_ucb1_score(c_param)
        )

    def get_ucb1_score(self, c_param=1.4) -> float:
        """
        Calculate the UCB1 score for this node.

        :param c_param: Exploration parameter.
        :return: The UCB1 score as a float.
        """
        return self.reward / self.visits + c_param * math.sqrt(math.log(self.available) / self.visits)

    def most_visited_child(self):
        """
        Get the child node with the highest visit count.

        This is typically used to select the final action after the search is complete.

        :return: The child node with the most visits.
        """
        return max(self.children.values(), key=lambda node: node.visits)

    def create_random_child(self) -> ISMCTSNode:
        """
        Expand the node by creating a new child node with a random unexplored action.

        :return: The newly created child node.
        """
        # Get legal actions from the current state
        action = random.choice(self._state.get_legal_actions())
        self.children[action] = ISMCTSNode(self, action)
        return self.children[action]

    def get_available_siblings(self) -> list[ISMCTSNode]:
        """
        Retrieve a list of sibling nodes that are available for selection.

        :return: A list of sibling ISMCTSNode instances.
        """
        return [c for c in self.parent.children.values() if c.action in self.parent._state.get_legal_actions() and c.action != self.action]

    def set_state(self, state: MCTSGameState):
        """
        Associate a game state with this node.

        :param state: The MCTSGameState to set.
        """
        self._state = state

    @staticmethod
    def get_random_determinization(obs: GameObservation) -> MCTSGameState:
        """
        Generate a random determinization of the game state consistent with the observation.

        In games with imperfect information, determinization is used to handle uncertainty
        by sampling possible game states.

        :param obs: The game observation containing known information.
        :return: A new MCTSGameState based on the determinized hands.
        """
        hands = get_random_determinization(obs)
        return mcts_state_from_observation(obs, hands)


class ISMCTS:
    """
    Implements the Information Set Monte Carlo Tree Search (ISMCTS) algorithm for imperfect information games.

    This class orchestrates the MCTS process, handling the tree search iterations,
    and ultimately selecting the best action to take from the current game observation.
    """
    def __init__(self, obs: GameObservation, rule: GameRule, iterations: int = 1000):
        """
        Initialize the ISMCTS algorithm.

        :param obs: The current game observation available to the player.
        :param rule: The game rule object defining valid moves and game logic.
        :param iterations: The number of iterations to perform in the search.
        """
        random.seed(1)                  # Set seed for reproducibility
        self.iterations = iterations    # Number of search iterations
        self.obs = obs                  # Current game observation
        self.root = ISMCTSNode()        # Root node of the search tree

    def search(self):
        """
        Execute the ISMCTS algorithm to find the best action from the current observation.

        :return: The action (card index) corresponding to the most promising move.
        """
        det = []
        alg = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            # Step 1: Determinization - sample a possible complete game state
            state = self.root.get_random_determinization(self.obs)
            self.root.set_state(state)
            determinization = time.perf_counter()

            # Step 2: Selection - traverse the tree to find a node to expand
            node, state = self.selection(self.root, state)

            # Step 3: Expansion - add a new child node if the node is not fully expanded
            if not node.is_fully_expanded(state) and not state.is_terminal:
                node, state = self.expand(node, state)

            # Step 4: Simulation - simulate a random playout from the node
            reward = self.simulation(state)

            # Step 5: Backpropagation - update the nodes with the simulation result
            self.backpropagate(node, reward)
            stop = time.perf_counter()
            det.append(determinization - start)
            alg.append(stop - determinization)

        print(f"{sum(det) / len(det):.6f} | {max(det):.6f} | {min(det):.6f} ||--|| {sum(alg) / len(alg):.6f} | {max(alg):.6f} | {min(alg):.6f}")

        # After all iterations, select the action with the most visits at the root
        return self.root.most_visited_child().action

    @staticmethod
    def selection(node: ISMCTSNode, state: MCTSGameState) -> tuple[ISMCTSNode, MCTSGameState]:
        """
        Traverse the tree, selecting child nodes based on UCB1 until a leaf node is reached.

        :param node: The current node in the tree.
        :param state: The current game state corresponding to the node.
        :return: A tuple containing the selected node and the updated game state.
        """
        while not state.is_terminal and node.is_fully_expanded(state):
            node = node.best_child()
            state = state.perform_action(node.action)
            node.set_state(state)
        return node, state

    @staticmethod
    def expand(node, state) -> tuple[ISMCTSNode, MCTSGameState]:
        """
        Expand the current node by adding a new child node for an unexplored action.

        :param node: The node to expand.
        :param state: The current game state.
        :return: A tuple containing the new child node and the updated game state.
        """
        child_node = node.create_random_child()
        child_state = state.perform_action(child_node.action)
        child_node.set_state(child_state)
        return child_node, child_state

    @staticmethod
    def simulation(state: MCTSGameState):
        """
        Simulate a random playout from the current state to a terminal state.

        :param state: The starting game state for the simulation.
        :return: The reward obtained at the end of the simulation.
        """
        state.run_internal_simulation()
        return state.get_reward()

    @staticmethod
    def backpropagate(node: ISMCTSNode, reward: float):
        """
        Backpropagate the simulation results up the tree, updating node statistics.

        :param node: The leaf node from which to start backpropagation.
        :param reward: The reward obtained from the simulation.
        """
        while node.parent is not None:
            node.visits += 1
            node.available += 1
            node.reward += reward
            for sibling in node.get_available_siblings():
                sibling.available += 1
            node = node.parent
