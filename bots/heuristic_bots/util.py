import numpy as np
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


def get_trump_selection_score(cards: np.ndarray, trump: int) -> int:
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


def get_better_card(card1: int, card2: int, trump: int) -> int:
    """
    Return the better card based on the current trump situation

    :param card1: Current best card
    :param card2: Next card in play
    :param trump: Current trump
    :return: Better card
    """
    if card2 == -1:
        return card1
    color1 = color_of_card[card1]
    color2 = color_of_card[card2]
    if color2 != color1:
        if color2 == trump:
            # If only later card is trump, later card is better
            return card2
        return card1
    # Both cards have same color
    color = color1
    if color == trump:
        # If both cards are trump, better trump card wins
        if higher_trump_card[card1 - 8 * color][card2 - 8 * color]:
            return card2
        return card1
    if trump == UNE_UFE:
        # If trump is UNE_UFE, lower card (larger constant value) wins
        return max(card1, card2)
    # All other scenarios are won by the higher card (smaller constant value)
    return min(card1, card2)


def get_current_best(trump: int, trick: np.ndarray) -> int:
    """
    Return the current best card in the trick

    # TODO fix docstring
    :return: Best card in the trick
    """
    best = trick[0]
    for card in trick[1:]:
        best = get_better_card(best, card, trump)
    return best


def get_remaining_trump_cards(trump: int, tricks: np.ndarray) -> list[int]:
    if trump in (OBE_ABE, UNE_UFE):
        return []
    return sorted(list(set(range(
            trump * 9,
            (trump + 1) * 9,
            1)
        ).difference(
            set(get_played_trump_cards(trump, tricks))
        )))


def get_played_trump_cards(trump: int, tricks: np.ndarray) -> list[int]:
    trumps_played = []
    if trump not in (OBE_ABE, UNE_UFE):
        for card in tricks.flatten():
            if card > -1 and color_of_card[card] == trump:
                trumps_played.append(card)
    return trumps_played


def get_bock_cards(cards: list[int], tricks: np.ndarray) -> list[int]:
    result = []
    for card in cards:
        if offset_of_card[card] > 0:
            for i in range(card - 1, card - offset_of_card[card] - 1, -1):
                if i not in tricks.flatten() and i not in cards:
                    break
            else:
                result.append(card)
        else:
            result.append(card)
    return result


def get_trump_cards(cards: list[int], trump: int) -> list[int]:
    if trump in (OBE_ABE, UNE_UFE):
        return []
    return [card for card in cards if color_of_card[card] == trump]


def get_points_in_trick(trump: int, trick: np.ndarray):
    return sum([card_values[trump][card] for card in trick if card > -1])