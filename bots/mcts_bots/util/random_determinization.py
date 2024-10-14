from __future__ import annotations

from jass.game.const import *
from jass.game.game_observation import GameObservation
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list


class RandomDeterminization:
    def __init__(self, obs: GameObservation):
        self._obs = obs
        self._hands = np.zeros([4, 36], np.int32)
        self._player_cards = [[], [], [], []]
        self._determine_possible_cards()
        self._determine_remaining_cards()
        self._determine_player_card_amount()
        self._allocate_cards_to_players()

    @property
    def hands(self):
        return self._hands

    def _determine_possible_cards(self):
        self._possible_cards_per_player = np.ones([4, 36], int)
        self._possible_cards_per_player[:, convert_one_hot_encoded_cards_to_int_encoded_list(self._obs.hand)] = 0
        self._possible_cards_per_player[self._obs.player, :] = 0

        # Remove all played cards and if applicable, color types from player's possible hands
        player = first_color = None
        for index, card in enumerate(self._obs.tricks[self._obs.tricks != -1]):
            self._possible_cards_per_player[:, card] = 0

            if not index % 4:  # First card in trick
                player = self._obs.trick_first_player[index // 4]
                first_color = color_of_card[card]
            else:
                player = next_player[player]
                color = color_of_card[card]

                if color != first_color and color != self._obs.trump:
                    offset = color_offset[first_color]
                    if (
                        first_color == self._obs.trump
                        and self._possible_cards_per_player[player, offset + J_offset] == 1
                    ):
                        self._possible_cards_per_player[player, offset: offset + 9] = 0
                        self._possible_cards_per_player[player, offset + J_offset] = 1
                    else:
                        self._possible_cards_per_player[player, offset: offset + 9] = 0

    def _determine_remaining_cards(self):
        cards = np.ones(36, np.int32)
        cards[self._obs.tricks[self._obs.tricks != -1]] = 0
        cards[convert_one_hot_encoded_cards_to_int_encoded_list(self._obs.hand)] = 0
        self._remaining_cards = {1: [], 2: [], 3: []}
        for card in convert_one_hot_encoded_cards_to_int_encoded_list(cards):
            players, = np.where(self._possible_cards_per_player[:, card] == 1)
            players = players.tolist()
            if not players:
                raise ValueError(f"Card cannot be assigned to any player: {card}")
            self._remaining_cards[len(players)].append((card, players))

    def _determine_player_card_amount(self):
        # How many cards each player must hold
        self._player_card_amount = [9 - self._obs.nr_tricks] * 4
        if self._obs.nr_cards_in_trick:
            for i in range(self._obs.nr_cards_in_trick):
                self._player_card_amount[(self._obs.player + 1 + i) % 4] -= 1
        self._player_card_amount[self._obs.player] = 0

    def _allocate_cards_to_players(self):
        card_amount_copy = list(self._player_card_amount)
        for i in (1, 2, 3):
            if not self._add_cards_with_n_possible_players(i):
                break
        else:
            self._hands[self._obs.player, :] = self._obs.hand
            for player, cards in enumerate(self._player_cards):
                self._hands[player, cards] = 1
            return
        self._player_card_amount = card_amount_copy
        self._player_cards = [[], [], [], []]
        self._allocate_cards_to_players()

    def _add_cards_with_n_possible_players(self, n_players: int) -> bool:
        for (card, players) in self._remaining_cards[n_players]:
            np.random.shuffle(players)
            for player in players:
                if self._player_card_amount[player] > 0:
                    self._player_cards[player].append(card)
                    self._player_card_amount[player] -= 1
                    break
            else:
                return False
        return True
