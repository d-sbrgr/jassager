from __future__ import annotations

import random

from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list


def get_random_determinization(obs: GameObservation) -> np.ndarray:
    possible_cards_per_player = _get_possible_cards(obs)
    remaining_cards = _get_remaining_cards(obs)
    player_card_amount = _get_player_card_amount(obs)

    hands = np.zeros([4, 36], np.int32)
    hands[obs.player, :] = obs.hand

    player_cards = _allocate_cards_to_players(remaining_cards, possible_cards_per_player, player_card_amount)

    for player, cards in enumerate(player_cards):
        hands[player, [cards]] = 1

    if obs.nr_played_cards + hands.sum() != 36:
        raise ValueError()

    return hands


def _get_possible_cards(obs: GameObservation) -> np.ndarray:
    possible_cards = np.ones([4, 36], int)
    # Remove all played cards and if applicable, color types from player's possible hands
    for index, card in enumerate(obs.tricks.flatten()):
        if card == -1:  # No more cards played
            break

        possible_cards[:, card] = 0

        if not index % 4:  # First card in trick
            player = obs.trick_first_player[index // 4]
            first_color = color_of_card[card]
            continue

        player = next_player[player]
        color = color_of_card[card]

        if color != first_color:
            if not color == obs.trump:
                offset = color_offset[first_color]
                if first_color == obs.trump and possible_cards[player, offset + J_offset] == 1:
                    possible_cards[player, offset: offset + 9] = 0
                    possible_cards[player, offset + J_offset] = 1
                else:
                    possible_cards[player, offset: offset + 9] = 0

    # Remove all cards in player_view's hand
    possible_cards[:, convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)] = 0
    possible_cards[obs.player, :] = 0  # Player_view's cards are already determined
    return possible_cards


def _get_remaining_cards(obs: GameObservation) -> list[int]:
    cards = np.ones(36, np.int32)
    for card in obs.tricks.flatten():
        if card == -1:
            break
        cards[card] = 0
    for card in convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand):
        cards[card] = 0
    return convert_one_hot_encoded_cards_to_int_encoded_list(cards)


def _get_player_card_amount(obs: GameObservation) -> list[int]:
    # How many cards each player must hold
    cards = [8 - obs.nr_tricks] * 4
    player = obs.player
    for i in range(4 - obs.nr_cards_in_trick):
        cards[player] += 1
        player = next_player[player]
    cards[obs.player] = 0
    return cards


def _allocate_cards_to_players(
        remaining_cards: list[int],
        possible_per_player: np.ndarray,
        card_amount: list[int]
) -> list[list[int]]:
    _check_no_cards_with_0_possible_players(remaining_cards, possible_per_player)
    remaining_copy = list(remaining_cards)
    card_amount_copy = list(card_amount)
    player_cards = [[], [], [], []]
    try:
        for i in (1, 2, 3):
            player_cards, remaining_cards, card_amount = _add_cards_with_n_possible_players(
                remaining_cards, player_cards, i, possible_per_player, card_amount
            )
    except ValueError:
        return _allocate_cards_to_players(remaining_copy, possible_per_player, card_amount_copy)

    return player_cards

def _add_cards_with_n_possible_players(
        cards: list[int],
        player_cards: list[list[int]],
        n_players: int,
        possible_per_player: np.ndarray,
        card_amount: list[int]
) -> tuple[list[list[int]], list[int], list[int]]:
    remaining_cards = []
    for card in cards:
        possible_players = possible_per_player[:, card]
        if possible_players.sum() == n_players:
            players, = np.where(possible_players == 1)
            players = players.tolist()
            random.shuffle(players)
            for player in players:
                if card_amount[player] > 0:
                    player_cards[player].append(card)
                    card_amount[player] -= 1
                    break
            else:
                raise ValueError()
        else:
            remaining_cards.append(card)
    return player_cards, remaining_cards, card_amount

def _check_no_cards_with_0_possible_players(
        cards: list[int],
        possible_per_player: np.ndarray
):
    for card in cards:
        possible_players = possible_per_player[:, card]
        if possible_players.sum() == 0:
            raise ValueError()
