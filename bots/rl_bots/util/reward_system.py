from jass.game.game_observation import GameObservation

def calculate_rewards(obs: GameObservation, immediate=False):
    """
    Calculate the reward for a given game observation.

    Parameters:
    - obs: The current game observation object.
    - immediate: Boolean flag for immediate rewards (default: False).

    Returns:
    - reward: Calculated reward for the observation.
    """
    if immediate:
        # Reward proportional to trick points
        team_points = obs.points[0]  # Adjust for the current player's team
        return team_points / 157  # Normalize by max points
    else:
        # Terminal reward based on total team score
        team_points = obs.points[0]
        opponent_points = obs.points[1]
        if team_points > opponent_points:
            return 1.0  # Win
        elif team_points < opponent_points:
            return -1.0  # Loss
        else:
            return 0.0  # Draw
