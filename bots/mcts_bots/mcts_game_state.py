from __future__ import annotations

import copy
import random

from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_rule import GameRule
from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, full_to_trump
from jass.game.rule_schieber import RuleSchieber


class MCTSGameState(GameState):
    _rule: GameRule = RuleSchieber()
    """
    Represents a game state within the Monte Carlo Tree Search (MCTS) framework.

    This class holds the current state of the game, provides methods to get legal actions,
    apply actions to generate new states, and calculate rewards.
    """

    def __init__(self) -> None:
        """
        Initialize the MCTS game state.
        """
        super().__init__()
        self.team = -1  # Team index (0 or 1)

    def get_reward(self) -> float:
        return float(self.points[self.team] / 157)

    # Get valid cards in one-hot encoded format and convert to integer list
    def get_legal_actions(self) -> list[int]:
        return convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_actions_from_state(self)
        )

    # Game ends when all 36 cards have been played
    @property
    def is_terminal(self) -> bool:
        return self.nr_played_cards == 36

    def run_internal_simulation(self):
        while not self.is_terminal:
            action = random.choice(self.get_legal_actions())
            self._action(action)

    def perform_action(self, action) -> MCTSGameState:
        """
        Apply an action to the current state and return the resulting new state.

        :param action: The action to perform (card index to play).
        :return: A new MCTSGameState reflecting the state after the action.
        """
        new_state = copy.deepcopy(self)
        new_state._action(action)
        return new_state

    def _action(self, action: int):
        if action < TRUMP_FULL_OFFSET:
            return self._action_play_card(action)
        trump_action = full_to_trump(action)
        return self._action_trump(trump_action)

    def _action_trump(self, action: int):
        if self.forehand == -1:
            # this is the action of the forehand player
            if action == PUSH:
                self.forehand = 0
                self.player = partner_player[self.player]
            else:
                self.forehand = 1
                self.trump = action
                self.declared_trump = self.player
                self.trick_first_player[0] = self.player
                # player remains the same
        elif self.forehand == 0:
            # action of the partner of the forehand player
            self.trump = action
            self.declared_trump = self.player
            self.player = next_player[self.dealer]
            self.trick_first_player[0] = self.player
        else:
            raise ValueError('Unexpected value {} for forehand in action_trump'.format(self.forehand))

    def _action_play_card(self, card: int):
        # remove card from player
        self.hands[self.player, card] = 0

        # place in trick
        self.current_trick[self.nr_cards_in_trick] = card
        self.nr_played_cards += 1

        if self.nr_cards_in_trick < 3:
            if self.nr_cards_in_trick == 0:
                # make sure the first player is set on the first card of a new trick
                # (it will not have been set, if the round has been restored from dict)
                self.trick_first_player[self.nr_tricks] = self.player
            # trick is not yet finished
            self.nr_cards_in_trick += 1
            self.player = next_player[self.player]
        else:
            points = self._rule.calc_points(self.current_trick, self.nr_played_cards == 36, self.trump)
            self.trick_points[self.nr_tricks] = points
            winner = self._rule.calc_winner(self.current_trick,
                                            self.trick_first_player[self.nr_tricks],
                                            self.trump)
            self.trick_winner[self.nr_tricks] = winner

            if winner == NORTH or winner == SOUTH:
                self.points[0] += points
            else:
                self.points[1] += points
            self.nr_tricks += 1
            self.nr_cards_in_trick = 0

            if self.nr_tricks < 9:
                # not end of round
                # next player is the winner of the trick
                self.trick_first_player[self.nr_tricks] = winner
                self.player = winner
                self.current_trick = self.tricks[self.nr_tricks, :]
            else:
                # end of round
                self.player = -1
                self.current_trick = None


def mcts_state_from_observation(obs: GameObservation, hands: np.ndarray) -> MCTSGameState:
    state = MCTSGameState()

    state.team = 0 if obs.player_view in (NORTH, SOUTH) else 1
    state.dealer = obs.dealer
    state.player = obs.player

    state.player_view = obs.player

    state.trump = obs.trump
    state.forehand = obs.forehand
    state.declared_trump = obs.declared_trump

    state.hands[:, :] = hands[:, :]

    state.tricks[:, :] = obs.tricks[:, :]
    state.trick_winner[:] = obs.trick_winner[:]
    state.trick_points[:] = obs.trick_points[:]
    state.trick_first_player[:] = obs.trick_first_player[:]
    state.nr_tricks = obs.nr_tricks
    state.nr_cards_in_trick = obs.nr_cards_in_trick

    # current trick is a view to the trick
    if obs.nr_played_cards < 36:
        state.current_trick = state.tricks[state.nr_tricks]
    else:
        state.current_trick = None

    state.nr_tricks = obs.nr_tricks
    state.nr_cards_in_trick = obs.nr_cards_in_trick
    state.nr_played_cards = obs.nr_played_cards
    state.points[:] = obs.points[:]

    return state