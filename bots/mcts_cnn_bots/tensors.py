import numpy as np
import pandas as pd
import torch

from jass.game.game_state import GameState
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.const import team, next_player
from jass.game.rule_schieber import RuleSchieber

RULE = RuleSchieber()


def get_played_cards_trick(s_: GameState) -> np.ndarray:
    result = np.zeros(36, dtype=np.float32)
    trick_value = 1 / 9
    for trick, cards in enumerate(s_.tricks, start=1):
        valid_cards = np.array(cards)
        stop_index = np.where(valid_cards == -1)[0]
        if stop_index.size > 0:
            valid_cards = valid_cards[:stop_index[0]]
        result[valid_cards] = trick * trick_value
    return result

def get_played_cards_player(s_: GameState) -> np.ndarray:
    result = np.zeros(36, dtype=np.int32)
    for trick, cards in enumerate(s_.tricks):
        player = s_.trick_first_player[trick]
        if player == -1:
            break
        valid_cards = np.array(cards)
        valid_cards = valid_cards[valid_cards != -1]  # Only process valid cards
        players = player - np.arange(len(valid_cards))  # Vectorized player calculation
        players %= len(next_player)  # Wrap around using modulo
        result[valid_cards] = players + 1
    return result


def game_state_to_cnn_tensor(s: GameState) -> torch.Tensor:
    tensor = torch.zeros((19, 4, 9))
    # Channel 0 - 3: Cards played (player ID)
    cards_player = get_played_cards_player(s)
    tensor[cards_player - 1, (np.arange(len(cards_player)) // 9), (np.arange(len(cards_player)) % 9)] = 1
    # Channel 4: Trick number
    cards_trick = torch.tensor(get_played_cards_trick(s))
    tensor[4] = cards_trick.reshape((4, 9))
    # Channel 5: Current player's hand
    t = tensor[5]
    t = t.flatten()
    t[convert_one_hot_encoded_cards_to_int_encoded_list(s.hands[s.player])] = 1
    # Channel 6: Valid cards to play
    t = tensor[6]
    t = t.flatten()
    t[convert_one_hot_encoded_cards_to_int_encoded_list(RULE.get_valid_actions_from_state(s))] = 1
    # Global information
    # Channel 7 - 12: Trump
    tensor[7 + s.trump, :, :] = 1
    # Channel 13: Current player
    tensor[13, :, :] = (s.player + 1) / 4
    # Channel 14: Declared trump
    tensor[14, :, :] = (s.declared_trump + 1) / 4
    # Channel 15: Forehand
    tensor[15, :, :] = s.forehand
    # Channel 16: Current points
    tensor[16, :, :] = s.points[team[s.player]] / 157
    # Channel 17: Total cards
    tensor[17, :, :] = s.nr_played_cards / 36
    # Channel 18: Played tricks
    tensor[18, :, :] = max(cards_trick)
    t = torch.zeros((1, 19, 4, 9))
    t[0] = tensor
    return t



def df_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = len(df)
    state_tensors = torch.zeros((num_samples, 19, 4, 9))
    policy_targets = torch.zeros((num_samples, 36), dtype=torch.long)
    value_targets = torch.zeros((num_samples, 1), dtype=torch.float32)

    for index, (_, row) in enumerate(df.iterrows()):
        # Encode the game state as a tensor
        state_tensor = game_state_to_tensor(
            trump=row['trump'],
            player=row['player'],
            declared_trump=row['declared_trump'],
            forehand=row['forehand'],
            hand=row['hand'],
            possible_actions=row['possible_actions'],
            cards_trick=row['cards_trick'],
            cards_player=row['cards_player'],
            current_points=row['current_points'],
            total_cards=row['total_cards'],
        )
        state_tensors[index] = state_tensor

        # Add policy and value targets
        pt = torch.zeros(36, dtype=torch.long)
        pt[row['card_played']] = 1
        policy_targets[index] = pt
        value_targets[index] = row['total_points']

    return state_tensors, policy_targets, value_targets


def game_state_to_tensor(
        trump: int,
        player: float,
        declared_trump: float,
        forehand: int,
        hand: list[int],
        possible_actions: list[int],
        cards_trick: list[float],
        cards_player: list[int],
        current_points: float,
        total_cards: float,
):
    num_channels = 19 # Channel per trump, player, declared_trump, forehand, hand, possible_actions, cards_trick, per cards_player, current_points, total_cards and total_tricks
    tensor = torch.zeros((num_channels, 4, 9))  # Initialize empty tensor

    # Channel 0 - 3: Cards played (player ID)
    cards_player = np.array(cards_player)
    tensor[cards_player - 1, (np.arange(len(cards_player)) // 9), (np.arange(len(cards_player)) % 9)] = 1

    # Channel 4: Trick number
    trick = torch.tensor(cards_trick).reshape((4, 9))
    tensor[4] = trick

    # Channel 5: Current player's hand
    t = tensor[5]
    t = t.flatten()
    t[hand] = 1

    # Channel 6: Valid cards to play
    t = tensor[6]
    t = t.flatten()
    t[possible_actions] = 1

    # Global information

    # Channel 7 - 12: Trump
    tensor[7 + trump, :, :] = 1

    # Channel 13: Current player
    tensor[13, :, :] = player

    # Channel 14: Declared trump
    tensor[14, :, :] = declared_trump

    # Channel 15: Forehand
    tensor[15, :, :] = forehand

    # Channel 16: Current points
    tensor[16, :, :] = current_points

    # Channel 17: Total cards
    tensor[17, :, :] = total_cards

    # Channel 18: Played tricks
    tensor[18, :, :] = max(cards_trick)

    return tensor