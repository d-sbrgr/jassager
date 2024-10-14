from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import full_to_trump

from .util.minimax import Minimax


class CheatingMinimax(AgentCheating):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        return full_to_trump(Minimax(state, 2).get_best_action())

    def action_play_card(self, state: GameState) -> int:
        return Minimax(state, 2).get_best_action()
