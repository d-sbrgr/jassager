# utils.py

import torch
import numpy as np
from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.arena.arena import Arena

from jass.game.const import *

from bots.rl_bots.util.jassnet import JassNet

def save_model(model, filepath="rl_model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class, filepath="rl_model.pth"):
    model = model_class(input_dim = 629, action_dim = 36)  # Instantiate the model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

def current_trick_can_win(current_trick: np.ndarray, player: int, trump: int) -> bool:
    """
    Determines whether the current player can win the ongoing trick.

    Parameters:
    - current_trick: Array representing the cards played in the current trick.
    - player: The index of the current player.
    - trump: The trump suit (-1 if no trump is active).

    Returns:
    - bool: True if the player can win the current trick, False otherwise.
    """
    # Filter out unplayed cards (-1 indicates no card played)
    played_cards = current_trick[current_trick >= 0]

    # If no cards are played yet, the player can win by default
    if len(played_cards) == 0:
        return True

    # Determine the highest card in the current trick
    # If the trick has trump cards, only trump cards are considered
    color_played = color_of_card[current_trick[0]]
    trump_cards = [card for card in played_cards if color_of_card[card] == trump]
    if trump_cards:
        # If trump cards are present, the highest trump wins
        highest_card = max(trump_cards, key=lambda card: lower_trump[card, :].sum())
    else:
        # Otherwise, the highest card of the leading color wins
        leading_color_cards = [card for card in played_cards if color_of_card[card] == color_played]
        highest_card = max(leading_color_cards, key=lambda card: card_values[trump, card])

    # Get the player's valid cards
    valid_cards = np.where(current_trick == -1)[0]  # Assuming valid_cards array is provided

    # Check if the player has a card that beats the current highest card
    for card in valid_cards:
        if (color_of_card[card] == trump and (not trump_cards or lower_trump[highest_card, card])) or \
           (color_of_card[card] == color_played and card < highest_card):
            return True

    return False

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


def get_card_values(cards: list[int], trump: int) -> list[int]:
    return list(map(lambda x: card_values[trump, x], cards))


def get_least_valuable_cards(cards: list[int], trump: int) -> list[int]:
    points = get_card_values(cards, trump)
    point_min = min(points)
    return [cards[i] for i in range(len(cards)) if points[i] == point_min]


def get_most_valuable_cards(cards: list[int], trump: int) -> list[int]:
    points = get_card_values(cards, trump)
    point_max = max(points)
    return [cards[i] for i in range(len(cards)) if points[i] == point_max]

