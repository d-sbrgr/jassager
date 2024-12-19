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
from jass.game.game_state import GameState

from jass.game.const import *

from bots.rl_bots.util.jassnet import JassNet

DEBUG = False

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


def get_better_card(card1, card2, trump):
    """
    Determine the better card between two cards given the trump suit.

    Args:
        card1 (int): Index of the first card (0-35).
        card2 (int): Index of the second card (0-35).
        trump (int): Trump suit (0-3 for suits, or special values for OBE_ABE, UNE_UFE).

    Returns:
        int: The index of the better card, or -1 if cards are incomparable.
    """
    # Calculate card colors
    color1 = card1 // 9
    color2 = card2 // 9

    # Compare different suits
    if color1 != color2:
        if color1 == trump:
            return card1
        elif color2 == trump:
            return card2
        else:
            return -1  # No direct comparison for non-trump cards of different suits

    # Validate cards
    assert 0 <= card1 < 36, f"Invalid card1: {card1}"
    assert 0 <= card2 < 36, f"Invalid card2: {card2}"

    # Relative card indices for comparison
    card1_relative = card1 - 9 * color1
    card2_relative = card2 - 9 * color2

    # Validate relative indices
    assert 0 <= card1_relative < 9, f"Invalid card1_relative: {card1_relative}, card1: {card1}, color1: {color1}"
    assert 0 <= card2_relative < 9, f"Invalid card2_relative: {card2_relative}, card2: {card2}, color2: {color2}"

    if DEBUG:
        print(f"card1: {card1}, card2: {card2}, trump: {trump}, color1: {color1}, color2: {color2}")
        print(f"card1_relative: {card1_relative}, card2_relative: {card2_relative}")

    # Comparison matrix for higher trump cards
    higher_trump_card = [
        [True, False, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False, False],
        [True, True, True, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False],
        [True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True, True, True],
    ]

    # Compare cards using the matrix
    if higher_trump_card[card1_relative][card2_relative]:
        return card1
    return card2




def get_current_best(trump, current_trick):
    """
    Get the best card in the current trick given the trump suit.

    Parameters:
    - trump: The trump suit (0-3 or special values for OBE_ABE/UNE_UFE).
    - current_trick: List of cards played in the current trick.

    Returns:
    - The index of the best card in the trick, or -1 if no cards have been played.
    """
    best_card = -1
    for card in current_trick:
        if card == -1:
            continue
        if best_card == -1:
            best_card = card
        else:
            best_card = get_better_card(best_card, card, trump)

    if DEBUG:
        print(f"trump: {trump}, current_trick: {current_trick}, best_card: {best_card}")

    return best_card



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


def get_bock_cards(cards: list[int], tricks: np.ndarray, trump: int) -> list[int]:
    """
    Identify Bock cards in the player's hand.

    Args:
        cards (list[int]): The list of cards in the player's hand.
        tricks (np.ndarray): The history of played tricks.
        trump (int): The current trump suit.

    Returns:
        list[int]: The list of Bock cards.
    """
    result = []

    # Skip Bock card logic for OBE_ABE and UNE_UFE
    if trump in [OBE_ABE, UNE_UFE]:
        return result

    for card in cards:
        # Skip trump cards
        if color_of_card[card] == trump:
            continue

        # Logic for identifying Bock cards
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

def is_same_trump_suit(action: int, current_best_card: int, trump: int) -> bool:
    """
    Check if the action is the same trump suit as the current_best_card.

    Args:
        action (int): The card to evaluate.
        current_best_card (int): The current best card in the trick.
        trump (int): The trump suit.

    Returns:
        bool: True if the action is the same trump suit as the current_best_card, False otherwise.
    """
    return color_of_card[action] == trump and color_of_card[current_best_card] == trump

def validate_game_state(state: GameState):
    """
    Validate the integrity of the GameState before encoding.

    Args:
    - state (GameState): The game state to validate.
    """
    if DEBUG:
        print(f"Validating state: player={state.player}, trump={state.trump}")
        print(f"state.hands.shape: {state.hands.shape}")
        print(f"state.current_trick: {state.current_trick}")
        print(f"state.trick_winner: {state.trick_winner}")

    assert state.hands.shape == (4, 36), f"Invalid hands shape: {state.hands.shape}"
    assert state.current_trick.shape == (4,), f"Invalid current_trick shape: {state.current_trick.shape}"
    assert state.tricks.shape == (9, 4), f"Invalid tricks shape: {state.tricks.shape}"
    assert len(state.points) == 2, f"Invalid points length: {len(state.points)}"
    assert 0 <= state.player < 4, f"Invalid player index: {state.player}"
    assert all(0 <= trick < 36 or trick == -1 for trick in state.current_trick), \
        f"Invalid values in current_trick: {state.current_trick}"
    assert all(0 <= winner < 4 or winner == -1 for winner in state.trick_winner), \
        f"Invalid values in trick_winner: {state.trick_winner}"


