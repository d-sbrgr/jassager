from jass.agents.agent import Agent
from jass.game.game_observation import GameObservation
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class RandomAgent(Agent):
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
        valid_actions = self._rule.get_valid_actions_from_obs(obs)
        # we use the global random number generator here
        return full_to_trump(np.random.choice(np.flatnonzero(valid_actions)))

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