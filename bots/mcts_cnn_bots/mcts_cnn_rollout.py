from jass.game.game_util import full_to_trump, convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

from bots.mcts_bots.util.mcts_implementation import ISMCTS
from bots.mcts_bots.util.mcts_game_state import DNNMCTSGameState, PureMCTSGameState
from models import load_model, JassCNN
from .tensors import game_state_to_cnn_tensor

MODEL = load_model(JassCNN, "jass_cnn", 1)


class MCTSCNNRollout(Agent):
    def __init__(self, c_param=1.3):
        self._rule = RuleSchieber()
        self._c_param = c_param

    def action_trump(self, obs: GameObservation) -> int:
        return int(full_to_trump(ISMCTS(obs, PureMCTSGameState).search()))

    def action_play_card(self, obs: GameObservation) -> int:
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs))
        if len(valid_moves) == 1:
            return int(valid_moves[0])
        return int(ISMCTS(obs, DNNMCTSGameState, self._c_param, MODEL, game_state_to_cnn_tensor).search())