from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from .heuristic_trump_random_play import HeuristicTrumpRandomPlay
from . import util


class FullHeuristicEgocentric(HeuristicTrumpRandomPlay):
    def action_play_card(self, obs: GameObservation) -> int:
        my_valid_cards = convert_one_hot_encoded_cards_to_int_encoded_list(self._rule.get_valid_cards_from_obs(obs))
        my_bock_cards = util.get_bock_cards(my_valid_cards, obs.tricks)
        my_trump_cards = util.get_trump_cards(my_valid_cards, obs.trump)
        # any bock card playable, play bock card
        if len(my_bock_cards) > 0:
            return np.random.choice(my_bock_cards)
        if obs.nr_cards_in_trick > 0:  # Play backhand
            if len(my_trump_cards) > 0:
                return np.random.choice(my_trump_cards)
        else:  # Play forehand
            if len(cards := set(my_valid_cards).difference(my_trump_cards)) > 0:
                return np.random.choice(list(cards))
        return np.random.choice(my_valid_cards)
