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
    vector_size = (4 * 36) + (4 * 36) + 6 + 4 + 2 + (9 * 36) + 4 + 1  # Dynamic size calculation
    encoded_state = np.zeros(vector_size, dtype=np.float32)

    # Modular offset tracking
    offset = 0

    # 1. Encode hands (36 cards per player, 4 players)
    offset = encode_hands(encoded_state, state, offset)

    # 2. Encode current trick and play order (36 cards per position in the trick)
    offset = encode_current_trick(encoded_state, state, offset)

    # 3. Encode trump (6 possible trumps: clubs, spades, hearts, diamonds, UNE_UFE, OBE_ABE)
    offset = encode_trump(encoded_state, state, offset)

    # 4. Encode current player position (4 positions: 0, 1, 2, 3)
    offset = encode_player_position(encoded_state, state, offset)

    # 5. Encode current scores (2 values: team 0, team 1)
    offset = encode_scores(encoded_state, state, offset)

    # 6. Encode trick history (9 tricks, 36 cards per trick)
    offset = encode_trick_history(encoded_state, state, offset)

    # 7. Encode trick counts (4 values: tricks won per player)
    offset = encode_trick_counts(encoded_state, state, offset)

    # 8. Encode additional features (e.g., forehand indicator)
    offset = encode_additional_features(encoded_state, state, offset)

    # Assert the final vector size matches expectations
    assert offset == vector_size, f"Encoded vector size mismatch: expected {vector_size}, got {offset}"

    return encoded_state


def encode_hands(encoded_state, state, offset):
    """
    Encode the hand cards for all 4 players.
    """
    for i, hand in enumerate(state.hands):
        encoded_state[offset:offset + 36] = hand
        offset += 36
    return offset


def encode_current_trick(encoded_state, state, offset):
    """
    Encode the current trick (36 cards per position in the trick).
    """
    for i, card in enumerate(state.current_trick):
        if card != -1:
            encoded_state[offset + card] = 1  # Mark the card played in this position
        offset += 36  # Move to the next position in the trick
    return offset


def encode_trump(encoded_state, state, offset):
    """
    Encode the trump suit (6 possible values: clubs, spades, hearts, diamonds, UNE_UFE, OBE_ABE).
    """
    if state.trump != -1:
        encoded_state[offset + state.trump] = 1  # Mark the selected trump
    offset += 6
    return offset


def encode_player_position(encoded_state, state, offset):
    """
    Encode the current player position (4 positions: 0, 1, 2, 3).
    """
    encoded_state[offset + state.player] = 1
    offset += 4
    return offset


def encode_scores(encoded_state, state, offset):
    """
    Encode the current scores for both teams.
    """
    # Normalize scores by the maximum possible score in a game (157)
    encoded_state[offset:offset + 2] = state.points / 157.0
    offset += 2
    return offset


def encode_trick_history(encoded_state, state, offset):
    """
    Encode the trick history (9 tricks, 36 cards per trick).
    """
    for trick in state.tricks:
        for card in trick:
            if card != -1:
                encoded_state[offset + card] = 1  # Mark the card played in this trick
        offset += 36  # Move to the next trick
    return offset


def encode_trick_counts(encoded_state, state, offset):
    """
    Encode the number of tricks won by each player.
    """
    # Count the number of tricks won by each player dynamically
    nr_tricks_per_player = [0, 0, 0, 0]
    for winner in state.trick_winner:
        if winner != -1:  # Only consider completed tricks
            nr_tricks_per_player[winner] += 1

    # Normalize trick counts by the total number of tricks in a game (9)
    encoded_state[offset:offset + 4] = np.array(nr_tricks_per_player) / 9.0
    offset += 4
    return offset



def encode_additional_features(encoded_state, state, offset):
    """
    Encode additional features such as forehand indicator.
    """
    # Example: Forehand indicator (1 if the current player is forehand, else 0)
    encoded_state[offset] = 1 if state.forehand == state.player else 0
    offset += 1
    return offset
