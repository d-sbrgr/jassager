import numpy as np
import pandas as pd
import torch


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