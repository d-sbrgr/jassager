# data_preprocessor.py

import pandas as pd
import numpy as np
from jass.game.const import * # Import card IDs from const.py
from jass.game.game_util import *

class DataPreprocessor:
    def __init__(self):
        self.index_to_card = card_ids
        self.suits = ['D', 'H', 'S', 'C']
        self.ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6']
        self.feature_columns = self._initialize_feature_columns()

    def _initialize_feature_columns(self):
        # List of feature columns for the model input, including engineered features
        base_columns = [card for card in card_ids.keys()]  # Base one-hot encoded cards
        engineered_columns = [
            'D_J9', 'D_AKQ', 'D_678', 'D_J_and_4_more', 'D_9A_and_3_more', 'D_J9A', 'D_J9_plus_card_and_2_other_aces',
            'H_J9', 'H_AKQ', 'H_678', 'H_J_and_4_more', 'H_9A_and_3_more', 'H_J9A', 'H_J9_plus_card_and_2_other_aces',
            'S_J9', 'S_AKQ', 'S_678', 'S_J_and_4_more', 'S_9A_and_3_more', 'S_J9A', 'S_J9_plus_card_and_2_other_aces',
            'C_J9', 'C_AKQ', 'C_678', 'C_J_and_4_more', 'C_9A_and_3_more', 'C_J9A', 'C_J9_plus_card_and_2_other_aces',
            'any_specified_feature_true',
        ]
        forehand_column = ['FH']
        
        return base_columns + forehand_column + engineered_columns

    def preprocess_hand(self, player_hand_array):
        """
        Converts a one-hot encoded player hand into a processed format suitable for model input.
        """


        hand_dict = {k: bool(v) for k, v in zip(card_strings, player_hand_array.flatten())}




        # hand_dict = {card: False for card in card_ids.keys()}
        # for index, has_card in enumerate(player_hand_array):
        #     card_name = self.index_to_card.get(index)
        #     hand_dict[card_name] = has_card

        # Convert to DataFrame
        hand_df = pd.DataFrame([hand_dict])
        hand_df = self._apply_feature_engineering(hand_df)
        hand_df = hand_df.reindex(columns=self.feature_columns, fill_value=False)

        return hand_df # Return as numpy array for model input

    def _count_suits(self, df):
        suits = {}
        for suit in 'DHSC':
            # Count all cards of the given suit (e.g., all Diamonds, Hearts, etc.)
            suits[suit] = df[[f'{suit}{rank}' for rank in ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6']]].sum(
                axis=1)
        return suits

    def _apply_feature_engineering(self, df):
        specified_features = []
        for suit in self.suits:
            # Jack and nine combination
            new_col = f'{suit}_J9'
            df[new_col] = df[f'{suit}J'] & df[f'{suit}9']
            specified_features.append(new_col)

            # A K Q combination
            new_col = f'{suit}_AKQ'
            df[new_col] = df[f'{suit}A'] & df[f'{suit}K'] & df[f'{suit}Q']
            specified_features.append(new_col)

            # 6 7 8 combination
            new_col = f'{suit}_678'
            df[new_col] = df[f'{suit}6'] & df[f'{suit}7'] & df[f'{suit}8']
            specified_features.append(new_col)

            # Jack and nine combination with 4 more cards
            new_col = f'{suit}_J_and_4_more'
            has_jack = df[f'{suit}J']
            suit_count = self._count_suits(df)[suit]
            df[new_col] = has_jack & (suit_count >= 5)
            specified_features.append(new_col)

            # 9, A, and at least 3 other cards in the same suit
            new_col = f'{suit}_9A_and_3_more'
            has_nine = df[f'{suit}9']
            has_ace = df[f'{suit}A']
            df[new_col] = has_nine & has_ace & (suit_count >= 5)
            specified_features.append(new_col)

            # Jack, 9, and Ace in the same suit
            new_col = f'{suit}_J9A'
            has_jack = df[f'{suit}J']
            has_nine = df[f'{suit}9']
            has_ace = df[f'{suit}A']
            df[new_col] = has_jack & has_nine & has_ace
            specified_features.append(new_col)

            # Jack, 9, and at least one other card in suit, plus two Aces in other suits
            new_col = f'{suit}_J9_plus_card_and_2_other_aces'
            other_ranks = ['A', 'K', 'Q', '10', '8', '7', '6']
            other_cards_in_suit = df[[f'{suit}{rank}' for rank in other_ranks]].any(axis=1)
            other_suits = [c for c in 'DHSC' if c != suit]
            aces_in_other_suits = df[[f'{c}A' for c in other_suits]].sum(axis=1)
            has_two_aces_in_other_suits = aces_in_other_suits >= 2
            df[new_col] = has_jack & has_nine & other_cards_in_suit & has_two_aces_in_other_suits
            specified_features.append(new_col)

        # Create a new column that is True if any of the specified features is True
        df['any_specified_feature_true'] = df[specified_features].any(axis=1)
        return df
