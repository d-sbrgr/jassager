from jass.game.game_util import full_to_trump, convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

from bots.mcts_bots.util.mcts_implementation_nn import ISMCTSNN


class FullMCTSNN(Agent):
    def __init__(self):
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        return full_to_trump(ISMCTSNN(obs).search())

    def action_play_card(self, obs: GameObservation) -> int:
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        if len(valid_moves) == 1:
            return valid_moves[0]
        return ISMCTSNN(obs).search()