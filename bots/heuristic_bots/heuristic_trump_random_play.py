from jass.agents.agent import Agent
from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber

from .util import get_trump_selection_score


class HeuristicTrumpRandomPlay(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        selection_scores = [get_trump_selection_score(cards, i) for i in range(6)]
        max_score = max(selection_scores)
        if max_score < 68 and obs.forehand == -1:
            return PUSH
        else:
            return selection_scores.index(max_score)

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(valid_cards))