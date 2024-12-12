import numpy as np

from jass.game.game_state import GameState
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber

rule_schieber = RuleSchieber()

def calculate_rewards_state(state: GameState, immediate=False):
    """
    Reward system incorporating Jass tactics and strategies for GameState.

    Parameters:
    - state: GameState object with details of the current game state.
    - immediate: Boolean flag for immediate rewards (default: False).

    Returns:
    - reward: Calculated reward for the given state.
    """
    try:
        # Extracting key info
        team_points = state.points[0]
        opponent_points = state.points[1]
        current_trick = state.current_trick
        hand = state.hands[state.player]
        trump = state.trump
        valid_moves = rule_schieber.get_valid_cards(hand, current_trick, state.nr_cards_in_trick, trump)

        reward = 0

        if immediate:
            # Valid move check
            if valid_moves.any():
                if np.sum(hand * valid_moves) > 0:
                    reward += 0.05
                else:
                    reward -= 0.2

            # Check if the player won the trick
            if state.nr_cards_in_trick == 3:
                winner = rule_schieber.calc_winner(current_trick, state.trick_first_player[state.nr_tricks], trump)
                if winner == state.player:
                    trick_points = rule_schieber.calc_points(current_trick, state.nr_played_cards == 36, trump)
                    reward += trick_points / 157.0
                    if state.nr_played_cards == 36:
                        reward += 0.1

            # Discourage wasting high-value cards
            high_value_cards = np.argwhere(hand * valid_moves > 0).flatten()
            if len(high_value_cards) > 0 and state.nr_cards_in_trick != 3:
                reward -= 0.05
        else:
            # Terminal reward
            margin = team_points - opponent_points
            reward = margin / 157.0
            if margin > 0:
                reward += 0.2 * (margin / 157.0)
            elif margin < 0:
                reward -= 0.2 * abs(margin / 157.0)

    except Exception as e:
        print(f"Error in calculate_rewards: {e}")

    return reward




def calculate_rewards_obs(obs: GameObservation, immediate=False):
    """
    Reward system incorporating Jass tactics and strategies.

    Parameters:
    - obs: GameObservation object with details of the current game state.
    - immediate: Boolean flag for immediate rewards (default: False).

    Returns:
    - reward: Calculated reward for the given observation.
    """
    # Extract key info from observation
    team_points = obs.points[0]
    opponent_points = obs.points[1]
    current_trick = obs.current_trick
    hand = obs.hand
    trump = obs.trump
    valid_moves = rule_schieber.get_valid_cards_from_obs(obs)

    if immediate:
        reward = 0

        # Check if the last move was valid
        if np.sum(hand * valid_moves) > 0:
            reward += 0.05  # Bonus for valid moves
        else:
            reward -= 0.2  # Penalty for invalid moves

        # Reward for winning the current trick
        if obs.nr_cards_in_trick == 3:  # Last card in the trick
            winner = rule_schieber.calc_winner(current_trick, obs.player, trump)
            if winner == obs.player_view:
                trick_points = rule_schieber.calc_points(current_trick, obs.nr_played_cards == 36, trump)
                reward += trick_points / 157  # Normalize by max points
                if obs.nr_played_cards == 36:  # Bonus for securing the last trick
                    reward += 0.1

        # Discourage wasting high-value cards unnecessarily
        high_value_cards = np.argwhere(hand * valid_moves > 0).flatten()
        if len(high_value_cards) > 0 and obs.nr_cards_in_trick != 3:
            # Penalize if a high-value card is played but doesn't win the trick
            reward -= 0.05

        return reward
    else:
        # Terminal reward based on the margin of victory
        margin = team_points - opponent_points
        reward = margin / 157  # Normalize reward

        # Reward scaling for strong victories
        if margin > 0:
            reward += 0.2 * (margin / 157)  # Reward larger margins more heavily
        elif margin < 0:
            reward -= 0.2 * abs(margin / 157)  # Penalize larger losses

        return reward

