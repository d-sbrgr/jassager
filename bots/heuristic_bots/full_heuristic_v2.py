from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from .heuristic_trump_random_play import HeuristicTrumpRandomPlay
from . import util


class FullHeuristicTableView(HeuristicTrumpRandomPlay):
    POINTS_THRESHOLD = 5

    def action_play_card(self, obs: GameObservation) -> int:
        my_valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(self._rule.get_valid_cards_from_obs(obs))
        my_bock_cards = util.get_bock_cards(my_valid_cards, obs.tricks)
        my_trump_cards = util.get_trump_cards(my_valid_cards, obs.trump)
        my_card_values = util.get_card_values(my_valid_cards, obs.trump)
        points_on_table = util.get_points_in_trick(obs.trump, obs.current_trick)
        opponent_trump_cards = util.get_remaining_trump_cards(obs.trump, obs.tricks)

        if obs.nr_cards_in_trick > 0:  # Play backhand
            if points_on_table > self.POINTS_THRESHOLD:
                if len(my_bock_cards) > 0:  # Points on table, Agent got bock
                    if len(opponent_trump_cards):  # Possible opponent trump play
                        return util.get_least_valuable_cards(my_bock_cards, obs.trump)[0]  # Play least valuable bock card
                    return util.get_most_valuable_cards(my_bock_cards, obs.trump)[0]  # Play most valuable bock card
                if len(my_trump_cards) > 0:
                    if len(opponent_trump_cards) > 0:  # Possible opponent trump play
                        return util.get_least_valuable_cards(my_trump_cards, obs.trump)[0]
                    return util.get_most_valuable_cards(my_trump_cards, obs.trump)[0]
            return util.get_least_valuable_cards(my_valid_cards, obs.trump)[0]
        else:  # Play forehand
            if len(my_trump_cards) > 1 and len(opponent_trump_cards) > 2:
                return util.get_most_valuable_cards(my_trump_cards, obs.trump)[0]
            if len(my_bock_cards) > 0:
                return util.get_most_valuable_cards(my_bock_cards, obs.trump)[0]
        return util.get_least_valuable_cards(my_valid_cards, obs.trump)[0]
