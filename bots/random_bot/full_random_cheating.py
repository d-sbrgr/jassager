from jass.agents.agent_cheating import AgentCheating
from jass.game.game_state import GameState
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber


class RandomAgentCheating(AgentCheating):
    def __init__(self):
        super().__init__()
        # We need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action for the given game state.

        Args:
            state: The game state. It must be in a state for trump selection.

        Returns:
            Selected trump as encoded in jass.game.const or jass.game.const.PUSH.
        """
        valid_actions = self._rule.get_valid_actions_from_state(state)
        # Convert full action indices to trump actions and pick randomly
        return int(full_to_trump(np.random.choice(np.flatnonzero(valid_actions))))

    def action_play_card(self, state: GameState) -> int:
        """
        Determine the card to play based on the game state.

        Args:
            state: The game state.

        Returns:
            The card to play, int encoded as defined in jass.game.const.
        """
        valid_cards = self._rule.get_valid_cards_from_state(state)
        # Pick a random valid card
        return int(np.random.choice(np.flatnonzero(valid_cards)))
