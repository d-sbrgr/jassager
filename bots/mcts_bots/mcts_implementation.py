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
from .util import get_random_determinization


class MCTSGameState:
    """
    Represents a game state within the Monte Carlo Tree Search (MCTS) framework.

    This class holds the current state of the game, provides methods to get legal actions,
    apply actions to generate new states, and calculate rewards.
    """
    def __init__(self, state: GameState, sim: GameSim, team: int) -> None:
        """
        Initialize the MCTS game state.

        :param state: The current game state.
        :param sim: The game simulator used to apply actions and progress the game.
        :param team: The index of the team (0 or 1) for which the reward is calculated.
        """
        self._simulation = sim  # Game simulator instance
        self._state = state     # Current game state
        self._team = team       # Team index (0 or 1)


    def get_reward(self) -> float:
        return float(self._state.points[self._team] / 157)

    # Get valid cards in one-hot encoded format and convert to integer list
    def get_legal_actions(self) -> list[int]:
        return convert_one_hot_encoded_cards_to_int_encoded_list(
            self._simulation.rule.get_valid_cards_from_state(self._state))

    # Game ends when all 36 cards have been played
    @property
    def is_terminal(self) -> bool:
        return self._state.nr_played_cards == 36

    def perform_action(self, action) -> MCTSGameState:
        """
        Apply an action to the current state and return the resulting new state.

        :param action: The action to perform (card index to play).
        :return: A new MCTSGameState reflecting the state after the action.
        """
        # Initialize the simulation from the current state
        self._simulation.init_from_state(self._state)
        # Perform the action in the simulation
        self._simulation.action(action)
        # Create and return a new MCTSGameState with the updated state
        return MCTSGameState(self._simulation.state, self._simulation, self._team)


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
        return all(legal_action in self.children.keys() for legal_action in state.get_legal_actions())

    def best_child(self, c_param=1.4) -> ISMCTSNode:
        """
        Select the best child node based on the Upper Confidence Bound (UCB1) score.

        :param c_param: Exploration parameter balancing exploitation and exploration.
        :return: The child node with the highest UCB1 score.
        """
        children = list(self.children.values())
        return max(
            children,
            key=lambda child: child.get_ucb1_score()
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
        # Create a new child node with the chosen action
        child = ISMCTSNode(self, action)
        # Add the new child to the children dictionary
        self.children[action] = child
        return child

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
    def get_random_determinization(obs: GameObservation, sim: GameSim) -> MCTSGameState:
        """
        Generate a random determinization of the game state consistent with the observation.

        In games with imperfect information, determinization is used to handle uncertainty
        by sampling possible game states.

        :param obs: The game observation containing known information.
        :param sim: The game simulator.
        :return: A new MCTSGameState based on the determinized hands.
        """
        hands = get_random_determinization(obs)
        return MCTSGameState(state_from_observation(obs, hands), sim, 0 if obs.player_view in (NORTH, SOUTH) else 1)


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
        self.sim = GameSim(rule)        # Game simulator initialized with the game rules
        self.root = ISMCTSNode()        # Root node of the search tree

    def search(self):
        """
        Execute the ISMCTS algorithm to find the best action from the current observation.

        :return: The action (card index) corresponding to the most promising move.
        """
        for _ in range(self.iterations):
            # Step 1: Determinization - sample a possible complete game state
            state = self.root.get_random_determinization(self.obs, self.sim)
            self.root.set_state(state)

            # Step 2: Selection - traverse the tree to find a node to expand
            node, state = self.selection(self.root, state)

            # Step 3: Expansion - add a new child node if the node is not fully expanded
            if not node.is_fully_expanded(state) and not state.is_terminal:
                node, state = self.expand(node, state)

            # Step 4: Simulation - simulate a random playout from the node
            reward = self.simulation(state)

            # Step 5: Backpropagation - update the nodes with the simulation result
            self.backpropagate(node, reward)

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
        while not state.is_terminal:
            action = random.choice(state.get_legal_actions())
            state = state.perform_action(action)
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
