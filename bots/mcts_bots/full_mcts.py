from jass.game.game_util import full_to_trump, convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation

from bots.mcts_bots.util.mcts_implementation import ISMCTS
from .heuristic_trump_mcts_play import HeuristicTrumpMCTSPlay

TIME_PER_REMAINING_CARD = 7/36


class FullMCTS(HeuristicTrumpMCTSPlay):
    def action_trump(self, obs: GameObservation) -> int:
        return full_to_trump(ISMCTS(obs).search())

    def action_play_card(self, obs: GameObservation) -> int:
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        if len(valid_moves) == 1:
            return valid_moves[0]
        return ISMCTS(obs, max_time=9).search()