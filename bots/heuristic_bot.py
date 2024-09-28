from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena


# Score for each card of a color from Ace to 6

trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]


def have_puur(hand: np.ndarray) -> np.ndarray:
    result = np.zeros(4, np.int32)
    result[0] = hand[DJ]
    result[1] = hand[HJ]
    result[2] = hand[SJ]
    result[3] = hand[CJ]
    return result


def have_puur_with_four(hand: np.ndarray) -> np.ndarray:
    result = np.zeros(4, dtype=int)
    colors = count_colors(hand)
    puurs = have_puur(hand)
    for i in range(4):
        result[i] = 1 if colors[i] >= 4 and puurs[i] > 0 else 0
    return result


def calculate_trump_selection_score(cards, trump: int) -> int:
    result = 0
    for card in cards:
        offset = offset_of_card[card]
        if trump == OBE_ABE:
            result += obenabe_score[offset]
        elif trump == UNE_UFE:
            result += uneufe_score[offset]
        else:
            color = color_of_card[card]
            if color == trump:
                result += trump_score[offset]
            else:
                result += no_trump_score[offset]
    return result


class HeuristicAgent(Agent):
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
        selection_scores = [calculate_trump_selection_score(cards, i) for i in range(6)]
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