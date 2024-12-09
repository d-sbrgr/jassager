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
    # Adjusted vector size (additional slots for 2 more trumps)
    vector_size = 481  # 479 + 2 for UNE_UFE and OBE_ABE
    encoded_state = np.zeros(vector_size, dtype=np.float32)

    # 1. Hand cards (36 cards, one-hot encoding)
    encoded_state[:36] = obs.hand

    # 2. Current trick (3 cards max, one-hot encoding)
    for i, card in enumerate(obs.current_trick):
        if card != -1:
            encoded_state[36 + i * 36 + card] = 1

    # 3. Trump information (6 trumps: 4 suits + UNE_UFE + OBE_ABE)
    trump_offset = 36 + 108
    if obs.trump != -1:
        encoded_state[trump_offset + obs.trump] = 1

    # 4. Player position (4 players, one-hot encoding)
    player_offset = trump_offset + 6
    encoded_state[player_offset + obs.player] = 1

    # 5. Scores (2 values)
    score_offset = player_offset + 4
    encoded_state[score_offset:score_offset + 2] = obs.points / 157.0  # Normalize scores

    # 6. Trick history (9 tricks max, one-hot encoding for each card)
    trick_offset = score_offset + 2
    for i in range(obs.nr_tricks):
        for j, card in enumerate(obs.tricks[i]):
            if card != -1:
                encoded_state[trick_offset + i * 36 + card] = 1

    return encoded_state
