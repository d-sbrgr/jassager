import numpy as np
from jass.game.game_state import GameState

# Debug flag
DEBUG = False


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

    # Validate the integrity of the GameState
    validate_game_state(state)

    # Modular offset tracking
    offset = 0

    # 1. Encode hands (36 cards per player, 4 players)
    offset = encode_hands(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding hands: {offset}")

    # 2. Encode current trick and play order (36 cards per position in the trick)
    offset = encode_current_trick(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding current trick: {offset}")

    # 3. Encode trump (6 possible trumps: clubs, spades, hearts, diamonds, UNE_UFE, OBE_ABE)
    offset = encode_trump(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding trump: {offset}")

    # 4. Encode current player position (4 positions: 0, 1, 2, 3)
    offset = encode_player_position(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding player position: {offset}")

    # 5. Encode current scores (2 values: team 0, team 1)
    offset = encode_scores(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding scores: {offset}")

    # 6. Encode trick history (9 tricks, 36 cards per trick)
    offset = encode_trick_history(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding trick history: {offset}")

    # 7. Encode trick counts (4 values: tricks won per player)
    offset = encode_trick_counts(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding trick counts: {offset}")

    # 8. Encode trick-level features (e.g., summary of tricks played)
    offset = encode_trick_level_features(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding trick-level features: {offset}")

    # 9. Encode additional features (e.g., forehand indicator)
    offset = encode_additional_features(encoded_state, state, offset)
    if DEBUG: print(f"Offset after encoding additional features: {offset}")

    # Assert the final vector size matches expectations
    assert offset == vector_size, f"Encoded vector size mismatch: expected {vector_size}, got {offset}"

    return encoded_state


def validate_game_state(state: GameState):
    """
    Validate the integrity of the GameState before encoding.

    Args:
    - state (GameState): The game state to validate.
    """
    assert state.hands.shape == (4, 36), f"Invalid hands shape: {state.hands.shape}"
    assert state.current_trick.shape == (4,), f"Invalid current_trick shape: {state.current_trick.shape}"
    assert state.tricks.shape == (9, 4), f"Invalid tricks shape: {state.tricks.shape}"
    assert len(state.points) == 2, f"Invalid points length: {len(state.points)}"
    assert 0 <= state.player < 4, f"Invalid player index: {state.player}"
    assert all(0 <= trick < 36 or trick == -1 for trick in state.current_trick), \
        f"Invalid values in current_trick: {state.current_trick}"
    assert all(0 <= winner < 4 or winner == -1 for winner in state.trick_winner), \
        f"Invalid values in trick_winner: {state.trick_winner}"


def encode_hands(encoded_state, state, offset):
    """
    Encode the hand cards for all 4 players.
    """
    for i, hand in enumerate(state.hands):
        assert len(hand) == 36, f"Invalid hand length for player {i}: {len(hand)}"
        encoded_state[offset:offset + 36] = hand
        offset += 36
    return offset


def encode_current_trick(encoded_state, state, offset):
    """
    Encode the current trick (36 cards per position in the trick).
    """
    for card in state.current_trick:
        if card != -1:  # Skip unplayed slots
            assert 0 <= card < 36, f"Invalid card in current trick: {card}"
            encoded_state[offset + card] = 1
    offset += 36
    return offset


def encode_trump(encoded_state, state, offset):
    """
    Encode the trump suit (6 possible values: clubs, spades, hearts, diamonds, UNE_UFE, OBE_ABE).
    """
    if state.trump != -1:
        assert 0 <= state.trump < 6, f"Invalid trump: {state.trump}"
        encoded_state[offset + state.trump] = 1
    offset += 6
    return offset


def encode_player_position(encoded_state, state, offset):
    """
    Encode the current player position (4 positions: 0, 1, 2, 3).
    """
    assert 0 <= state.player < 4, f"Invalid player index: {state.player}"
    encoded_state[offset + state.player] = 1
    offset += 4
    return offset


def encode_scores(encoded_state, state, offset):
    """
    Encode the current scores for both teams.
    """
    assert state.points is not None, "State points are None"
    assert len(state.points) == 2, f"State points have invalid length: {len(state.points)}"
    encoded_state[offset:offset + 2] = state.points / 157.0  # Normalize scores
    offset += 2
    return offset


def encode_trick_history(encoded_state, state, offset):
    """
    Encode the trick history (9 tricks, 36 cards per trick).
    """
    for trick in state.tricks.flatten():
        if trick != -1:
            assert 0 <= trick < 36, f"Invalid card in trick history: {trick}"
            encoded_state[offset + trick] = 1
    offset += 9 * 36
    return offset



def encode_trick_counts(encoded_state, state, offset):
    """
    Encode the number of tricks won by each player.
    """
    nr_tricks_per_player = [0, 0, 0, 0]
    for winner in state.trick_winner:
        if winner != -1:
            assert 0 <= winner < 4, f"Invalid trick winner index: {winner}"
            nr_tricks_per_player[winner] += 1
    encoded_state[offset:offset + 4] = np.array(nr_tricks_per_player) / 9.0  # Normalize by total tricks
    offset += 4
    return offset


def encode_trick_level_features(encoded_state, state, offset):
    """
    Encode trick-level features to fill the missing 108 slots.
    """
    encoded_state[offset:offset + 108] = 0  # Example placeholder, replace with actual data if needed
    offset += 108
    return offset


def encode_additional_features(encoded_state, state, offset):
    """
    Encode additional features such as forehand indicator.
    """
    encoded_state[offset] = 1 if state.forehand == state.player else 0
    offset += 1
    return offset
