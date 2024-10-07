from jass.game.game_observation import GameObservation

from .util import ISMCTS
from ..heuristic_bots.heuristic_trump_random_play import HeuristicTrumpRandomPlay


class HeuristicTrumpMCTSPlay(HeuristicTrumpRandomPlay):
    def action_play_card(self, obs: GameObservation) -> int:
        if len(valid_moves := self._rule.get_valid_cards_from_obs(obs)) == 1:
            return valid_moves[0]
        return ISMCTS(root_state=initial_state, player=player, iterations=1000).search()