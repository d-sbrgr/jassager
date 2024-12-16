from jass.game.game_state import GameState
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.game.const import *
from bots.rl_bots.util.utils import *
import numpy as np

# Initialize RuleSchieber
rule = RuleSchieber()

# Debug flag
debug = False


def calculate_rewards_state(state: GameState, immediate=False, action=None) -> float:
    """
    Calculate rewards based on the current GameState.

    Args:
        state (GameState): The current game state.
        immediate (bool): Flag for intermediate rewards.
        action (int): The action taken (for trump rewards or card play).

    Returns:
        float: The calculated reward.
    """
    reward = 0.0

    if state.trump == -1:  # Trump selection phase
        reward += calculate_trump_rewards(state.hands[state.player], action)

    elif immediate:
        if state.nr_cards_in_trick == 4:  # Trick completed
            reward += calculate_trick_rewards(state)
        elif action is not None:  # Card play phase
            reward += calculate_card_play_rewards(state, action)

    if state.nr_played_cards == 36:  # Game completed
        reward += calculate_terminal_rewards(state)

    return reward


def calculate_rewards_obs(obs: GameObservation, immediate=False, action=None) -> float:
    """
    Calculate rewards based on the current GameObservation.

    Args:
        obs (GameObservation): The current game observation.
        immediate (bool): Flag for intermediate rewards.
        action (int): The action taken (for trump rewards or card play).

    Returns:
        float: The calculated reward.
    """
    reward = 0.0

    if obs.trump == -1:  # Trump selection phase
        reward += calculate_trump_rewards(obs.hand, action)

    elif immediate:
        if obs.nr_cards_in_trick == 4:  # Trick completed
            reward += calculate_trick_rewards(obs)
        elif action is not None:  # Card play phase
            reward += calculate_card_play_rewards(obs, action)

    if obs.nr_played_cards == 36:  # Game completed
        reward += calculate_terminal_rewards(obs)

    return reward


def calculate_trump_rewards(hand: np.ndarray, action: int) -> float:
    """
    Calculate rewards for trump selection with teamwork considerations.

    Args:
        hand (np.ndarray): One-hot encoded hand of the player.
        action (int): The selected trump action.

    Returns:
        float: The calculated trump reward.
    """
    if action not in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]:
        return -1.0  # Penalty for invalid trump selection

    # Reward for maximizing "Puur" cards in the selected trump suit
    trump_puur_bonus = have_puur_with_four(hand)[action] * 0.5

    # Reward for choosing a trump that maximizes utility
    trump_selection_score = get_trump_selection_score(hand, action)
    trump_scores = [get_trump_selection_score(hand, trump) for trump in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]]
    max_score, min_score = max(trump_scores), min(trump_scores)

    # Reward for picking the trump with the highest utility score
    trump_score_bonus = 0.5 if trump_selection_score == max_score else -0.5 if trump_selection_score == min_score else 0.1

    # Combine teamwork and utility rewards
    trump_reward = 0.1 + trump_puur_bonus + trump_score_bonus

    if debug:
        print(f"Trump selection: {action}, Puur bonus: {trump_puur_bonus}, "
              f"Score bonus: {trump_score_bonus}, Total trump reward: {trump_reward}")

    return trump_reward


def calculate_card_play_rewards(state_or_obs, action: int) -> float:
    """
    Calculate rewards for card play.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The selected card action.

    Returns:
        float: The calculated card play reward.
    """
    valid_moves = (
        rule.get_valid_cards_from_state(state_or_obs)
        if isinstance(state_or_obs, GameState)
        else rule.get_valid_cards_from_obs(state_or_obs)
    )
    reward = 0.0

    # Reward for playing a valid card
    if valid_moves[action]:
        reward += 0.1

        # Reward for playing a Bock card
        hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs, GameState) else state_or_obs.hand
        tricks = state_or_obs.tricks
        trump = state_or_obs.trump
        reward += calculate_bock_rewards(hand, action, tricks, trump) # See function for reward

        # Bonus for winning the trick with the card
        if state_or_obs.nr_cards_in_trick == 4:
            reward += calculate_trick_rewards(state_or_obs)

        # Add penalty for wasting valuable cards
        reward += calculate_avoid_wasting_card_reward(state_or_obs, action)

        # Add rewards for handling special suits (OBE_ABE and UNE_UFE)
        reward += calculate_special_suit_card_reward(state_or_obs, action)

        # Add penalty for unnecessary point loss
        reward += calculate_avoid_point_loss_reward(state_or_obs, action)

    else:
        # Penalty for invalid card
        reward -= 0.2

    return reward



def calculate_trick_rewards(state_or_obs) -> float:
    """
    Calculate rewards for winning or losing a trick.

    Args:
        state_or_obs: GameState or GameObservation.

    Returns:
        float: The calculated trick reward.
    """
    winner = state_or_obs.trick_winner[state_or_obs.nr_tricks - 1]
    player = state_or_obs.player if isinstance(state_or_obs, GameObservation) else state_or_obs.player
    return 0.5 if winner == player else -0.5


def calculate_terminal_rewards(state_or_obs) -> float:
    """
    Calculate terminal rewards at the end of the game.

    Args:
        state_or_obs: GameState or GameObservation.

    Returns:
        float: The calculated terminal reward.
    """
    team_points = state_or_obs.points[0]
    opponent_points = state_or_obs.points[1]
    reward = 0.0

    # Base rewards for winning or losing
    if team_points > opponent_points:
        reward += 1 * (team_points / 100)
    elif team_points < opponent_points:
        reward -= 1

    # Baseline reward/penalty based on trump chooser
    trump_chooser = get_trump_chooser(state_or_obs)
    if trump_chooser in [1, 3]:  # Opponent chose trump
        if team_points < 50:
            reward -= 1.0  # Penalty for underperformance
        elif team_points > 50:
            reward += 1.0  # Reward for outperforming
    elif trump_chooser in [0, 2]:  # Agent or teammate chose trump
        if team_points < 100:
            reward -= 1.0  # Penalty for underperformance
        elif team_points > 100:
            reward += 1.0  # Reward for outperforming

    if debug:
        print(f"Terminal reward: {reward}, Team points: {team_points}, Opponent points: {opponent_points}, "
              f"Trump chooser: {trump_chooser}")

    return reward

def get_trump_chooser(state_or_obs) -> int:
    """
    Derive the trump chooser (player who declared trump) from the state or observation.

    Args:
        state_or_obs: GameState or GameObservation.

    Returns:
        int: The player index (0-3) who declared trump.
    """
    forehand = state_or_obs.forehand
    dealer = state_or_obs.dealer

    if forehand == 1:  # Forehand: First player after the dealer
        return next_player[dealer]
    elif forehand == 0:  # Rearhand: Dealer's partner
        return partner_player[next_player[dealer]]
    return -1

from bots.rl_bots.util.utils import get_bock_cards

def calculate_bock_rewards(hand, played_card, tricks, trump) -> float:
    """
    Calculate rewards for playing a Bock card.

    Args:
        hand (np.ndarray): The current hand of the player.
        played_card (int): The card played by the agent.
        tricks (np.ndarray): The history of played tricks.
        trump (int): The current trump suit.

    Returns:
        float: The calculated Bock reward.
    """
    bock_cards = get_bock_cards(hand, tricks)
    if played_card in bock_cards:
        return 0.5  # Reward for playing a Bock card
    return 0.0

def calculate_avoid_wasting_card_reward(state_or_obs, action: int) -> float:
    """
    Penalize the agent for wasting high-value cards unnecessarily.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        float: Reward (or penalty) for avoiding wasting valuable cards.
    """
    # Determine the current best card in the trick
    current_best_card = get_current_best(state_or_obs.trump, state_or_obs.current_trick)

    # If the agent's card can't beat the current best card, check its value
    if action != current_best_card and not can_card_win(action, current_best_card, state_or_obs.trump):
        card_value = get_card_values([action], state_or_obs.trump)[0]

        # Penalize based on card value (higher penalty for higher-value cards)
        if card_value >= 10:  # High-value cards (e.g., Ace, King, 10)
            return -0.5
        elif card_value >= 5:  # Medium-value cards (e.g., 9, Jack)
            return -0.2

    # No penalty if the card can win the trick or is a low-value card
    return 0.0


def can_card_win(card: int, current_best_card: int, trump: int) -> bool:
    """
    Determine if a card can win against the current best card.

    Args:
        card (int): The card to evaluate.
        current_best_card (int): The current best card in the trick.
        trump (int): The trump suit.

    Returns:
        bool: True if the card can win, False otherwise.
    """
    if color_of_card[card] == trump and color_of_card[current_best_card] != trump:
        return True  # Trump beats non-trump
    if color_of_card[card] == color_of_card[current_best_card]:
        # Compare ranks within the same suit
        return card < current_best_card
    return False

def calculate_special_suit_card_reward(state_or_obs, action: int) -> float:
    """
    Reward the agent for playing the right card in Obenabe or Uneufe.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        float: Reward for aligning with Obenabe or Uneufe rules.
    """
    trump = state_or_obs.trump

    # Check if the game is in Obenabe or Uneufe mode
    if trump == OBE_ABE:  # Obenabe: highest card wins
        # Reward for playing the highest available card
        if is_highest_card_available(state_or_obs, action):
            return 0.3
    elif trump == UNE_UFE:  # Uneufe: lowest card wins
        # Reward for playing the lowest available card
        if is_lowest_card_available(state_or_obs, action):
            return 0.3

    return 0.0


def is_highest_card_available(state_or_obs, action: int) -> bool:
    """
    Check if the played card is the highest available card.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        bool: True if the card is the highest available, False otherwise.
    """
    hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs, GameState) else state_or_obs.hand
    available_cards = np.flatnonzero(hand)
    return action == max(available_cards)


def is_lowest_card_available(state_or_obs, action: int) -> bool:
    """
    Check if the played card is the lowest available card.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        bool: True if the card is the lowest available, False otherwise.
    """
    hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs, GameState) else state_or_obs.hand
    available_cards = np.flatnonzero(hand)
    return action == min(available_cards)

def calculate_avoid_point_loss_reward(state_or_obs, action: int) -> float:
    """
    Penalize the agent for playing a card that unnecessarily gives points to opponents.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        float: Penalty for unnecessary point loss.
    """
    trump = state_or_obs.trump
    current_best_card = get_current_best(trump, state_or_obs.current_trick)

    # Get the points of the played card
    card_points = get_card_values([action], trump)[0]

    # Check if the card gives points unnecessarily
    if action > current_best_card and card_points > 0:
        return -0.3  # Penalty for unnecessarily losing points

    return 0.0

