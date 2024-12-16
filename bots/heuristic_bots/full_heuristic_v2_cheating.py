from jass.game.game_state import GameState
from jass.game.game_util import *
from bots.heuristic_bots.heuristic_trump_random_play_cheating import HeuristicTrumpRandomPlayCheating
from . import util


class FullHeuristicTableViewCheating(HeuristicTrumpRandomPlayCheating):
    POINTS_THRESHOLD = 5

    def action_play_card(self, state: GameState) -> int:
        """
        Determine the card to play based on the game state.

        Args:
            state: The current game state.

        Returns:
            The card to play, int encoded as defined in jass.game.const.
        """
        current_hand = state.hands[state.player]
        valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards(current_hand, state.current_trick, state.nr_cards_in_trick, state.trump)
        )

        # Extract card groups and values
        my_bock_cards = util.get_bock_cards(valid_cards, state.tricks)
        my_trump_cards = util.get_trump_cards(valid_cards, state.trump)
        my_card_values = util.get_card_values(valid_cards, state.trump)
        points_on_table = util.get_points_in_trick(state.trump, state.current_trick)
        opponent_trump_cards = util.get_remaining_trump_cards(state.trump, state.tricks)

        if state.nr_cards_in_trick > 0:  # Play backhand
            if points_on_table > self.POINTS_THRESHOLD:
                if len(my_bock_cards) > 0:  # Points on table, agent has bock
                    if len(opponent_trump_cards) > 0:  # Possible opponent trump play
                        # Play least valuable bock card
                        return util.get_least_valuable_cards(my_bock_cards, state.trump)[0]
                    # Play most valuable bock card
                    return util.get_most_valuable_cards(my_bock_cards, state.trump)[0]
                if len(my_trump_cards) > 0:
                    if len(opponent_trump_cards) > 0:  # Possible opponent trump play
                        # Play least valuable trump card
                        return util.get_least_valuable_cards(my_trump_cards, state.trump)[0]
                    # Play most valuable trump card
                    return util.get_most_valuable_cards(my_trump_cards, state.trump)[0]
            # Play least valuable card
            return util.get_least_valuable_cards(valid_cards, state.trump)[0]
        else:  # Play forehand
            if len(my_trump_cards) > 1 and len(opponent_trump_cards) > 2:
                # Play most valuable trump card
                return util.get_most_valuable_cards(my_trump_cards, state.trump)[0]
            if len(my_bock_cards) > 0:
                # Play most valuable bock card
                return util.get_most_valuable_cards(my_bock_cards, state.trump)[0]
        # Play least valuable card
        return util.get_least_valuable_cards(valid_cards, state.trump)[0]
