from jass.game.game_state import GameState
from bots.cheating.cheating_agents import RLAgentCheating
import numpy as np
import traceback

# Mock model (for testing logic)
class MockModel:
    def __call__(self, state):
        return np.zeros((1, 36)), 0  # Mock policy and value outputs

# Initialize GameState
state = GameState()
state.hands[0][0:9] = 1  # Assign first 9 cards to player 0
state.player = 0  # Player 0's turn
state.current_trick[0] = -1  # Current trick is empty
state.nr_cards_in_trick = 0
state.trump = 0  # Example trump

# Initialize RLAgentCheating
agent = RLAgentCheating(model=MockModel())





try:
    # Test action_play_card
    try:
        action = agent.action_play_card(state)
        print(f"Agent chose action: {action}")
    except Exception as e:
        print(f"Error during agent execution: {e}")

except AttributeError as e:
    print(f"Error: {e}")
    traceback.print_exc()
    raise
