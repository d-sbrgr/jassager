from __future__ import annotations

import random

from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list


class RandomDeterminization:
    def __init__(self, obs: GameObservation):
        self._obs = obs
        self._hands = np.zeros([4, 36], np.int32)
        self._player_cards = [[], [], [], []]
        self._possible_cards_per_player = self._get_possible_cards()
        self._remaining_cards = self._get_remaining_cards()
        self._player_card_amount = self._get_player_card_amount()
        self._allocate_cards_to_players()

    @property
    def hands(self):
        return self._hands

    def _get_possible_cards(self) -> np.ndarray:
        player = first_color = None
        possible_cards = np.ones([4, 36], int)
        # Remove all played cards and if applicable, color types from player's possible hands
        for index, card in enumerate(self._obs.tricks.flatten()):
            if card == -1:  # No more cards played
                break

            possible_cards[:, card] = 0

            if not index % 4:  # First card in trick
                player = self._obs.trick_first_player[index // 4]
                first_color = color_of_card[card]
                continue

            player = next_player[player]
            color = color_of_card[card]

            if color != first_color:
                if not color == self._obs.trump:
                    offset = color_offset[first_color]
                    if first_color == self._obs.trump and possible_cards[player, offset + J_offset] == 1:
                        possible_cards[player, offset: offset + 9] = 0
                        possible_cards[player, offset + J_offset] = 1
                    else:
                        possible_cards[player, offset: offset + 9] = 0

        # Remove all cards in player_view's hand
        possible_cards[:, convert_one_hot_encoded_cards_to_int_encoded_list(self._obs.hand)] = 0
        possible_cards[self._obs.player, :] = 0  # Player_view's cards are already determined
        return possible_cards

    def _get_remaining_cards(self) -> list[int]:
        cards = np.ones(36, np.int32)
        for card in self._obs.tricks.flatten():
            if card == -1:
                break
            cards[card] = 0
        for card in convert_one_hot_encoded_cards_to_int_encoded_list(self._obs.hand):
            cards[card] = 0
        return convert_one_hot_encoded_cards_to_int_encoded_list(cards)

    def _get_player_card_amount(self) -> list[int]:
        # How many cards each player must hold
        cards = [8 - self._obs.nr_tricks] * 4
        player = self._obs.player
        for i in range(4 - self._obs.nr_cards_in_trick):
            cards[player] += 1
            player = next_player[player]
        cards[self._obs.player] = 0
        return cards

    def _allocate_cards_to_players(self):
        self._check_no_cards_with_0_possible_players()
        remaining_copy = list(self._remaining_cards)
        card_amount_copy = list(self._player_card_amount)
        try:
            for i in (1, 2, 3):
                self._add_cards_with_n_possible_players(i)
        except ValueError:
            self._remaining_cards = remaining_copy
            self._player_card_amount = card_amount_copy
            self._player_cards = [[], [], [], []]
            self._allocate_cards_to_players()
        else:
            self._hands[self._obs.player, :] = self._obs.hand
            for player, cards in enumerate(self._player_cards):
                self._hands[player, cards] = 1

    def _add_cards_with_n_possible_players(self, n_players: int):
        remaining_cards, self._remaining_cards = self._remaining_cards, []
        for card in remaining_cards:
            possible_players = self._possible_cards_per_player[:, card]
            if possible_players.sum() == n_players:
                players, = np.where(possible_players == 1)
                players = players.tolist()
                random.shuffle(players)
                for player in players:
                    if self._player_card_amount[player] > 0:
                        self._player_cards[player].append(card)
                        self._player_card_amount[player] -= 1
                        break
                else:
                    raise ValueError()
            else:
                self._remaining_cards.append(card)

    def _check_no_cards_with_0_possible_players(
            self
    ):
        for card in self._remaining_cards:
            possible_players = self._possible_cards_per_player[:, card]
            if possible_players.sum() == 0:
                raise ValueError()
