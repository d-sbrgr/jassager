
from jass.game.game_state import GameState
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.game.const import *
from bots.rl_bots.util.utils import *
import numpy as np

# Initialize RuleSchieber
rule = RuleSchieber()

# DEBUG flag
DEBUG = True

# Constants for reward values
INVALID_TRUMP_SELECTION_PENALTY = -1.0
TRUMP_SELECTION_REWARD = 0.5
WEAK_TRUMP_SELECTION_PENALTY = -0.2
TRICK_REWARD = 0.5
TRICK_PENALTY = -0.5
GOOD_CARD_REWARD = 0.2
BAD_CARD_PENALTY = -0.2
INVALID_CARD_PENALTY = -0.2
VALID_CARD_REWARD = 0.1
BOCK_CARD_REWARD = 0.5
WASTING_CARD_PENALTY = -0.1
SPECIAL_SUIT_REWARD = 0.3
POINT_LOSS_PENALTY = -0.3

def calculate_rewards_state(state, immediate=False, action=None):
    """
    Calculate rewards for the current game state.

    Parameters:
    - state: The current game state.
    - immediate: Whether to calculate immediate rewards.
    - action: The action being evaluated.

    Returns:
    - The calculated reward value.
    """
    reward = 0.0

    try:
        if state.trump == -1:
            reward += calculate_trump_rewards(state.hands[state.player], action, state)

        # Calculate rewards for intermediate steps
        elif immediate:
            # action played
            if action is not None:
                reward += calculate_card_play_rewards(state, action)
        # Terminal rewards
        if state.nr_played_cards == 36:
            reward += calculate_terminal_rewards(state)
    except Exception as e:
        if DEBUG:
            print(f"Error during reward calculation: {e}")
            print(f"state: {state}, action: {action}")
        raise e

    if DEBUG:
        print(f"Final reward: {reward}, immediate={immediate}, action={action}")
    return reward

def calculate_trump_rewards(hand: np.ndarray, action: int, state_or_obs) -> float:
    """
    Calculate rewards for trump selection with teamwork considerations, including PUSH logic
    and rewards for pushing when trump selection score is very low.

    Args:
        hand (np.ndarray): One-hot encoded hand of the player.
        action (int): The selected trump action.
        state_or_obs: GameState or GameObservation for determining forehand.

    Returns:
        float: The calculated trump reward.
    """
    if action not in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE, PUSH]:
        return INVALID_TRUMP_SELECTION_PENALTY  # Penalty for invalid trump selection

    # Calculate trump scores for comparison
    trump_scores = [get_trump_selection_score(hand, trump) for trump in
                    [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]]
    min_score = min(trump_scores)
    max_score = max(trump_scores)

    # Special handling for PUSH
    if action == PUSH:
        if state_or_obs.forehand == 0:
            return INVALID_TRUMP_SELECTION_PENALTY  # Penalty if PUSH was already used

        if max_score < 68:  # Threshold for mediocre score
            return TRUMP_SELECTION_REWARD  # Reward for pushing with a low score

        if max_score > 80:
            return INVALID_TRUMP_SELECTION_PENALTY  # Penalty for selecting PUSH with a high score

    # Encourage valid trump selection
    trump_puur_bonus = 0
    if action in [DIAMONDS, HEARTS, SPADES, CLUBS]:
        # Check if the player has a jack and at least 3 additional cards in the same suit
        puurs = have_puur(hand)
        colors = count_colors(hand)
        if puurs[action] > 0 and colors[action] >= 4:
            trump_puur_bonus += TRUMP_SELECTION_REWARD  # Bonus reward for selecting a strong trump
        else:
            trump_puur_bonus += WEAK_TRUMP_SELECTION_PENALTY  # Slight penalty for weak trump selection

    # Reward for choosing a trump that maximizes utility
    trump_selection_score = get_trump_selection_score(hand, action)

    if trump_selection_score == max_score:
        trump_score_bonus = TRUMP_SELECTION_REWARD
    elif trump_selection_score == min_score:
        trump_score_bonus = WEAK_TRUMP_SELECTION_PENALTY
    else:
        normalized_score = (trump_selection_score - (max_score + min_score) / 2) / (
                    max_score - min_score)
        trump_score_bonus = normalized_score / 2

    # Combine teamwork and utility rewards
    trump_reward = trump_puur_bonus + trump_score_bonus

    if DEBUG:
        print(f"Trump selection: {action}, Puur bonus: {trump_puur_bonus}, "
              f"Score bonus: {trump_score_bonus}, Total trump reward: {trump_reward}, "
              f"Min trump score: {min_score}, Max trump score: {max_score}")

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
        reward += VALID_CARD_REWARD

        # Reward for playing a Bock card
        hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs,
                                                                     GameState) else state_or_obs.hand
        tricks = state_or_obs.tricks
        trump = state_or_obs.trump
        reward += calculate_bock_rewards(hand, action, tricks, trump)  # See function for reward

        # Bonus for winning the trick with the card
        if state_or_obs.nr_cards_in_trick == 3:
            # Simulate adding the agent's action to complete the trick
            simulated_trick = state_or_obs.current_trick.copy()
            simulated_trick[state_or_obs.nr_cards_in_trick] = action

            # Temporarily calculate the reward for the completed trick
            reward += calculate_trick_rewards(state_or_obs, simulated_trick)
            if DEBUG: print(f"Trickwinning reward?: {reward}")
        elif state_or_obs.nr_cards_in_trick == 4:
            # Directly calculate trick rewards when the trick is already full
            reward += calculate_trick_rewards(state_or_obs)

        # Add penalty for wasting valuable cards
        reward += calculate_avoid_wasting_card_reward(state_or_obs, action)

        # Add rewards for handling special suits (OBE_ABE and UNE_UFE)
        reward += calculate_special_suit_card_reward(state_or_obs, action)

        # Add penalty for unnecessary point loss
        reward += calculate_avoid_point_loss_reward(state_or_obs, action)

    else:
        # Penalty for invalid card
        reward += INVALID_CARD_PENALTY

    return reward

def calculate_trick_rewards(state_or_obs, simulated_trick=None) -> float:
    """
    Calculate rewards for winning or losing a trick.

    Args:
        state_or_obs: GameState or GameObservation.
        simulated_trick (np.ndarray, optional): Simulated trick to include agent's action.

    Returns:
        float: The calculated trick reward.
    """
    if simulated_trick is not None:
        trick = simulated_trick  # Use the simulated trick
    else:
        trick = state_or_obs.current_trick

    winner = rule.calc_winner(trick, state_or_obs.trick_first_player[state_or_obs.nr_tricks],
                              state_or_obs.trump)
    player = state_or_obs.player if isinstance(state_or_obs,
                                               GameObservation) else state_or_obs.player
    if DEBUG: print(f"Trickwinning reward?: {winner}")
    return TRICK_REWARD if winner == player else TRICK_PENALTY

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

    max_points = 157

    # Positive reward for winning, negative for losing
    if team_points > opponent_points:
        reward += (2 + ((team_points - opponent_points) / 100))
    elif team_points < opponent_points:
        reward -= (2 + ((opponent_points - team_points) / 100))

    # Baseline reward/penalty based on trump chooser
    trump_chooser = get_trump_chooser(state_or_obs)
    if trump_chooser in [1, 3]:  # Opponent chose trump
        if team_points < 50:
            reward -= 0.5  # Penalty for underperformance
        elif team_points > 50:
            reward += 0.5  # Reward for outperforming
    elif trump_chooser in [0, 2]:  # Agent or teammate chose trump
        if team_points < 100:
            reward -= 0.5  # Penalty for underperformance
        elif team_points > 100:
            reward += 0.5  # Reward for outperforming

    if DEBUG:
        print(
            f"Terminal reward: {reward}, Team points: {team_points}, Opponent points: {opponent_points}, "
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
    bock_cards = get_bock_cards(hand, tricks, trump)
    if played_card in bock_cards:
        return BOCK_CARD_REWARD  # Reward for playing a Bock card
    return 0.0

def calculate_avoid_wasting_card_reward(state_or_obs, action):
    """
    Calculate a reward to avoid wasting high-value cards unnecessarily.

    Parameters:
    - state_or_obs: The current game state or observation.
    - action: The card index (0-35) being played.

    Returns:
    - A reward value (e.g., float).
    """
    # Validate current_trick
    for card in state_or_obs.current_trick:
        assert card == -1 or 0 <= card < 36, f"Invalid card in current_trick: {card}"

    # Get the current best card in the trick
    current_best_card = get_current_best(state_or_obs.trump, state_or_obs.current_trick)

    # Calculate reward logic here (example placeholder)
    reward = 0.0
    if can_hand_win(state_or_obs.hands[state_or_obs.player], current_best_card, state_or_obs.trump):
        if current_best_card != -1:
            if is_same_trump_suit(action, current_best_card, state_or_obs.trump):
                if action > current_best_card:
                    reward += GOOD_CARD_REWARD  # reward playing higher cards strategically
                else:
                    reward += BAD_CARD_PENALTY  # Penalize wasting cards unnecessarily
            elif color_of_card[action] == color_of_card[current_best_card] and color_of_card[
                action] != state_or_obs.trump:
                if action > current_best_card:
                    reward += GOOD_CARD_REWARD  # reward playing higher cards strategically
                else:
                    reward += BAD_CARD_PENALTY  # Penalize wasting cards unnecessarily
    else:
        if is_lowest_card_available(state_or_obs, action):
            reward += GOOD_CARD_REWARD  # Reward for playing a low-value card
        elif action in get_bock_cards(state_or_obs.hands[state_or_obs.player], state_or_obs.tricks,
                                      state_or_obs.trump) or get_card_values([action], state_or_obs.trump)[0] > 3:
            reward += BAD_CARD_PENALTY  # Penalize for playing a high-value or Bock card


    if DEBUG:
        print(f"action: {action}, current_best_card: {current_best_card}, reward: {reward}")

    return reward

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
        if trump == UNE_UFE:
            return card > current_best_card
        elif color_of_card[card] != trump:
            return card < current_best_card
        else:
            return higher_trump_card[current_best_card, card] == 1
    return False

def calculate_special_suit_card_reward(state_or_obs, action: int) -> float:
    """
    Reward the agent for playing the overall highest (Obenabe) or lowest (Uneufe) card of the color,
    not just the highest/lowest in their hand.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        float: Reward for aligning with Obenabe or Uneufe rules.
    """
    trump = state_or_obs.trump
    current_trick = state_or_obs.current_trick
    hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs,
                                                                 GameState) else state_or_obs.hand

    # Extract color of the played card
    action_color = color_of_card[action]

    # Combine played cards and cards in the current trick
    played_cards = state_or_obs.tricks.flatten()
    all_played_cards = [card for card in played_cards if card != -1] + list(
        current_trick[current_trick != -1])

    # Identify all cards of the played card's color
    all_color_cards = [card for card in range(action_color * 9, (action_color + 1) * 9)]

    # Determine remaining cards in the color
    remaining_color_cards = list(set(all_color_cards) - set(all_played_cards))

    if trump == OBE_ABE:  # Obenabe: highest card wins
        highest_card = min(remaining_color_cards)  # Lowest index represents highest rank
        if action == highest_card:
            return GOOD_CARD_REWARD  # Reward for playing the overall highest card
        else:
            return BOCK_CARD_REWARD  # Penalty for not playing the overall highest card

    elif trump == UNE_UFE:  # Uneufe: lowest card wins
        lowest_card = max(remaining_color_cards)  # Highest index represents lowest rank
        if action == lowest_card:
            return GOOD_CARD_REWARD  # Reward for playing the overall lowest card
        else:
            return BOCK_CARD_REWARD  # Penalty for not playing the overall lowest card

    return 0.0  # No reward outside of special suits

def is_highest_card_available(state_or_obs, action: int) -> bool:
    """
    Check if the played card is the highest available card.

    Args:
        state_or_obs: GameState or GameObservation.
        action (int): The card played by the agent.

    Returns:
        bool: True if the card is the highest available, False otherwise.
    """
    hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs,
                                                                 GameState) else state_or_obs.hand
    available_cards = np.flatnonzero(hand) % 9
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
    hand = state_or_obs.hands[state_or_obs.player] if isinstance(state_or_obs,
                                                                 GameState) else state_or_obs.hand
    available_cards = np.flatnonzero(hand) % 9
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
    if action != current_best_card and card_points > 3:
        return BAD_CARD_PENALTY  # Penalty for unnecessarily losing points

    return 0.0


def can_hand_win(hand: np.ndarray, current_best_card: int, trump: int) -> bool:
    """
    Determine if the agent can win with the current cards in hand against the current_best_card.

    Args:
        hand (np.ndarray): The current hand of the player.
        current_best_card (int): The current best card in the trick.
        trump (int): The trump suit.

    Returns:
        bool: True if the agent can win with any card in hand, False otherwise.
    """
    for card in np.flatnonzero(hand):
        if can_card_win(card, current_best_card, trump):
            return True
    return False


