# encode_game_state.py

import numpy as np

from jass.game.game_state import GameState

def encode_game_state(state: GameState) -> np.ndarray:
    """
    Encodes the GameState object into a fixed-size input vector for the neural network.

    Parameters:
    - state (GameState): The current game state with perfect information.

    Returns:
    - np.ndarray: Encoded state vector.
    """
    # Adjusted vector size (additional slots for 2 more trumps)
    vector_size = 481  # 479 + 2 for UNE_UFE and OBE_ABE
    encoded_state = np.zeros(vector_size, dtype=np.float32)

    try:
        # 1. Hand cards for all players
        hand_offset = 0
        for i, hand in enumerate(state.hands):
            start_idx = hand_offset + i * 36
            end_idx = start_idx + 36
            if end_idx <= vector_size:
                encoded_state[start_idx:end_idx] = hand

        # 2. Current trick
        trick_offset = hand_offset + 4 * 36
        for i, card in enumerate(state.current_trick):
            if card != -1:
                trick_idx = trick_offset + i * 36 + card
                if trick_idx < vector_size:
                    encoded_state[trick_idx] = 1

        # 3. Trump information
        trump_offset = trick_offset + 4 * 36
        if state.trump != -1 and trump_offset + state.trump < vector_size:
            encoded_state[trump_offset + state.trump] = 1

        # 4. Player position
        player_offset = trump_offset + 6
        if player_offset + state.player < vector_size:
            encoded_state[player_offset + state.player] = 1

        # 5. Scores
        score_offset = player_offset + 4
        if score_offset + 2 <= vector_size:
            encoded_state[score_offset:score_offset + 2] = state.points / 157.0

        # 6. Trick history
        history_offset = score_offset + 2
        for i, trick in enumerate(state.tricks):
            for j, card in enumerate(trick):
                if card != -1:
                    history_idx = history_offset + i * 36 + card
                    if history_idx < vector_size:
                        encoded_state[history_idx] = 1

    except IndexError as e:
        print(f"IndexError in encode_game_state: {e}")

    return encoded_state
