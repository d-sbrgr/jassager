import numpy as np
from jass.game.game_observation import GameObservation

# Debug flag for validation purposes
DEBUG = False

def encode_game_observation(obs: GameObservation) -> np.ndarray:
    """
    Encodes the GameObservation object into a fixed-size input vector for the neural network.

    Parameters:
    - obs (GameObservation): The current game observation.

    Returns:
    - np.ndarray: Encoded state vector.
    """
    vector_size = (4 * 36) + (4 * 36) + 6 + 4 + 2 + (9 * 36) + 4 + 1  # Total = 629
    encoded_state = np.zeros(vector_size, dtype=np.float32)

    # Validate the integrity of the GameObservation
    validate_game_observation(obs)

    # Modular offset tracking
    offset = 0

    # 1. Encode player's hand
    offset = encode_hand(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding hand: {offset}")

    # 2. Encode placeholders for other players' hands
    offset = encode_empty_hands(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding empty hands: {offset}")

    # 3. Encode the current trick
    offset = encode_current_trick(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding current trick: {offset}")

    # 4. Encode trump
    offset = encode_trump(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding trump: {offset}")

    # 5. Encode player position
    offset = encode_player_position(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding player position: {offset}")

    # 6. Encode scores
    offset = encode_scores(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding scores: {offset}")

    # 7. Encode trick history
    offset = encode_trick_history(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding trick history: {offset}")

    # 8. Encode trick counts
    offset = encode_trick_counts(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding trick counts: {offset}")

    # 9. Encode additional features
    offset = encode_additional_features(encoded_state, obs, offset)
    if DEBUG: print(f"Offset after encoding additional features: {offset}")

    # Assert final offset consistency
    assert offset == vector_size, f"Encoded vector size mismatch: expected {vector_size}, got {offset}"
    return encoded_state


def validate_game_observation(obs: GameObservation):
    """
    Validate the integrity of the GameObservation before encoding.
    """
    assert len(obs.hand) <= 36, f"Invalid hand length: {len(obs.hand)}"
    assert isinstance(obs.current_trick, list), "Current trick must be a list"
    assert 0 <= obs.player < 4, f"Invalid player index: {obs.player}"
    assert 0 <= obs.trump < 6 or obs.trump == -1, f"Invalid trump value: {obs.trump}"


def encode_hand(encoded_state, obs, offset):
    """
    Encode the player's hand cards (36 slots).
    """
    for card in obs.hand:
        if 0 <= card < 36:
            encoded_state[offset + card] = 1
    offset += 36
    return offset


def encode_empty_hands(encoded_state, obs, offset):
    """
    Placeholder for the other players' hands (3 * 36 slots).
    """
    encoded_state[offset:offset + (3 * 36)] = 0
    offset += (3 * 36)
    return offset


def encode_current_trick(encoded_state, obs, offset):
    """
    Encode the current trick (36 cards per position).
    """
    for card in obs.current_trick:
        if 0 <= card < 36:
            encoded_state[offset + card] = 1
        offset += 36
    return offset


def encode_trump(encoded_state, obs, offset):
    """
    Encode the trump suit (6 slots).
    """
    if obs.trump != -1:
        encoded_state[offset + obs.trump] = 1
    offset += 6
    return offset


def encode_player_position(encoded_state, obs, offset):
    """
    Encode the current player position (4 slots).
    """
    encoded_state[offset + obs.player] = 1
    offset += 4
    return offset


def encode_scores(encoded_state, obs, offset):
    """
    Encode the scores for both teams (2 slots).
    """
    encoded_state[offset:offset + 2] = obs.points / 157.0  # Normalize scores
    offset += 2
    return offset


def encode_trick_history(encoded_state, obs, offset):
    """
    Encode the trick history (9 * 36 slots).
    """
    for trick in obs.tricks:
        for card in trick:
            if card != -1:
                encoded_state[offset + card] = 1
        offset += 36
    return offset


def encode_trick_counts(encoded_state, obs, offset):
    """
    Encode the number of tricks won by each player (4 slots).
    """
    nr_tricks_per_player = [0, 0, 0, 0]
    for winner in obs.trick_winner:
        if 0 <= winner < 4:
            nr_tricks_per_player[winner] += 1
    encoded_state[offset:offset + 4] = np.array(nr_tricks_per_player) / 9.0
    offset += 4
    return offset


def encode_additional_features(encoded_state, obs, offset):
    """
    Encode additional features such as forehand indicator (1 slot).
    """
    encoded_state[offset] = 1 if obs.forehand == obs.player else 0
    offset += 1
    return offset
