import numpy as np
from jass.game.game_observation import GameObservation

def encode_game_observation(obs: GameObservation) -> np.ndarray:
    """
    Encodes the GameObservation object into a fixed-size input vector for the neural network.

    Parameters:
    - obs (GameObservation): The current game observation.

    Returns:
    - np.ndarray: Encoded state vector.
    """
    # Adjusted vector size to match GameState encoding (629)
    vector_size = (4 * 36) + (4 * 36) + 6 + 4 + 2 + (9 * 36) + 4 + 1  # Total = 629
    encoded_state = np.zeros(vector_size, dtype=np.float32)

    offset = 0
    offset = encode_hand(encoded_state, obs, offset)
    offset = encode_empty_hands(encoded_state, obs, offset)  # Placeholder for other players' hands
    offset = encode_current_trick(encoded_state, obs, offset)
    offset = encode_trump(encoded_state, obs, offset)
    offset = encode_player_position(encoded_state, obs, offset)
    offset = encode_scores(encoded_state, obs, offset)
    offset = encode_trick_history(encoded_state, obs, offset)
    offset = encode_trick_counts(encoded_state, obs, offset)
    offset = encode_additional_features(encoded_state, obs, offset)

    # Validate final offset
    assert offset == vector_size, f"Encoded vector size mismatch: expected {vector_size}, got {offset}"

    return encoded_state


def encode_hand(encoded_state, obs, offset):
    """
    Encode the hand cards for the observing player.
    """
    encoded_state[offset:offset + 36] = obs.hand
    offset += 36
    return offset


def encode_empty_hands(encoded_state, obs, offset):
    """
    Placeholder for the other players' hands.
    """
    # Fill with zeros since the agent doesn't have access to other hands
    encoded_state[offset:offset + (3 * 36)] = 0
    offset += (3 * 36)
    return offset


def encode_current_trick(encoded_state, obs, offset):
    """
    Encode the current trick (36 cards per position in the trick).
    """
    for i, card in enumerate(obs.current_trick):
        if card != -1:
            encoded_state[offset + card] = 1  # Mark the card played in this position
        offset += 36  # Move to the next position in the trick
    return offset


def encode_trump(encoded_state, obs, offset):
    """
    Encode the trump suit (6 possible values: clubs, spades, hearts, diamonds, UNE_UFE, OBE_ABE).
    """
    if obs.trump != -1:
        encoded_state[offset + obs.trump] = 1  # Mark the selected trump
    offset += 6
    return offset


def encode_player_position(encoded_state, obs, offset):
    """
    Encode the current player position (4 positions: 0, 1, 2, 3).
    """
    encoded_state[offset + obs.player] = 1
    offset += 4
    return offset


def encode_scores(encoded_state, obs, offset):
    """
    Encode the current scores for both teams.
    """
    # Normalize scores by the maximum possible score in a game (157)
    encoded_state[offset:offset + 2] = obs.points / 157.0
    offset += 2
    return offset


def encode_trick_history(encoded_state, obs, offset):
    """
    Encode the trick history (9 tricks, 36 cards per trick).
    """
    for trick in obs.tricks:
        for card in trick:
            if card != -1:
                encoded_state[offset + card] = 1  # Mark the card played in this trick
        offset += 36  # Move to the next trick
    return offset


def encode_trick_counts(encoded_state, obs, offset):
    """
    Encode the number of tricks won by each player.
    """
    # Count the number of tricks won by each player dynamically
    nr_tricks_per_player = [0, 0, 0, 0]
    for winner in obs.trick_winner:
        if winner != -1:  # Only consider completed tricks
            nr_tricks_per_player[winner] += 1

    # Normalize trick counts by the total number of tricks in a game (9)
    encoded_state[offset:offset + 4] = np.array(nr_tricks_per_player) / 9.0
    offset += 4
    return offset


def encode_additional_features(encoded_state, obs, offset):
    """
    Encode additional features such as forehand indicator.
    """
    # Example: Forehand indicator (1 if the current player is forehand, else 0)
    encoded_state[offset] = 1 if obs.forehand == obs.player else 0
    offset += 1
    return offset
