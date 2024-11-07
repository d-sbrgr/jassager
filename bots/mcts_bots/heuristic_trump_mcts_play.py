from jass.game.game_observation import GameObservation

from bots.mcts_bots.util.mcts_implementation import ISMCTS
from ..heuristic_bots.heuristic_trump_random_play import HeuristicTrumpRandomPlay


class HeuristicTrumpMCTSPlay(HeuristicTrumpRandomPlay):
    def action_play_card(self, obs: GameObservation) -> int:
        if len(valid_moves := self._rule.get_valid_cards_from_obs(obs)) == 1:
            return int(valid_moves[0])
        return int(ISMCTS(obs).search())
