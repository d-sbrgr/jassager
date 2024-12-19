import numpy as np
from jass.agents.agent_cheating import AgentCheating
from jass.game.const import *
from jass.game.game_state import GameState
from jass.game.game_util import *
from jass.game.rule_schieber import RuleSchieber
from .util import get_trump_selection_score


class HeuristicTrumpRandomPlayCheating(AgentCheating):
    def __init__(self):
        super().__init__()
        # Initialize a RuleSchieber object for determining valid cards
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        """
        Determine trump action for the given game state.

        Args:
            state: The game state, which must be in a state for trump selection.

        Returns:
            Selected trump as encoded in jass.game.const or jass.game.const.PUSH.
        """
        current_hand = state.hands[state.player]
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(current_hand)
        selection_scores = [get_trump_selection_score(cards, i) for i in range(6)]
        max_score = max(selection_scores)
        if max_score < 68 and state.forehand == -1:  # Threshold to decide if PUSH is better
            return PUSH
        return selection_scores.index(max_score)

    def action_play_card(self, state: GameState) -> int:
        """
        Determine the card to play.

        Args:
            state: The current game state.

        Returns:
            The card to play, int encoded as defined in jass.game.const.
        """
        current_hand = state.hands[state.player]
        valid_cards = self._rule.get_valid_cards(current_hand, state.current_trick, state.nr_cards_in_trick,
                                                 state.trump)

        # Ensure there are valid cards to play
        if not valid_cards.any():
            raise ValueError("No valid cards available to play!")

        # Randomly select from valid cards
        return np.random.choice(np.flatnonzero(valid_cards))
