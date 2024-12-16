from jass.game.game_util import full_to_trump, convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from bots.mcts_bots.util.mcts_implementation import ISMCTS


class FullMCTS(Agent):
    def __init__(self):
        self._rule = RuleSchieber()
        self.experience_buffer = []  # Experience buffer for storing states, actions, rewards

    def action_trump(self, obs: GameObservation) -> int:
        # Validate observation
        assert obs.hand is not None, "obs.hand is missing"
        assert obs.trump == -1, "Trump has already been declared"

        # Perform ISMCTS search for trump
        search_result = ISMCTS(obs).search()
        assert search_result is not None, "ISMCTS search returned None for trump"

        # Store the experience (state and trump action)
        self.store_experience(obs, action=full_to_trump(search_result), reward=0)  # No immediate reward for trump
        return int(full_to_trump(search_result))

    def action_play_card(self, obs: GameObservation) -> int:
        # Validate observation
        assert obs.hand is not None, "obs.hand is missing"
        assert obs.current_trick is not None, "obs.current_trick is missing"
        assert len(obs.hand) > 0, "Player hand is empty"

        # Fetch valid moves
        valid_moves = convert_one_hot_encoded_cards_to_int_encoded_list(
            self._rule.get_valid_cards_from_obs(obs)
        )
        assert len(valid_moves) > 0, "No valid moves available"

        # Handle single valid move
        if len(valid_moves) == 1:
            action = int(valid_moves[0])
            self.store_experience(obs, action=action, reward=0)  # No immediate reward
            return action

        # Perform ISMCTS search
        search_result = ISMCTS(obs).search()
        assert search_result is not None, "ISMCTS search returned None for card action"

        # Store the experience (state, action, reward)
        self.store_experience(obs, action=search_result, reward=0)  # Reward will be calculated later
        return int(search_result)

    def store_experience(self, obs: GameObservation, action: int, reward: float):
        """
        Store a transition in the experience buffer.

        Args:
            obs (GameObservation): Current observation of the game.
            action (int): Action taken by the agent.
            reward (float): Reward for the action.
        """
        # Encode the current state
        from bots.rl_bots.util.encode_game_obs import encode_game_observation
        try:
            encoded_state = encode_game_observation(obs)
        except Exception as e:
            print(f"Error encoding observation: {e}")
            return

        # Add the experience to the buffer
        self.experience_buffer.append({
            "state": encoded_state.tolist(),
            "action": action,
            "reward": reward
        })

    def finalize_game(self, final_obs: GameObservation):
        """
        Finalize the game by assigning rewards to each transition.

        Args:
            final_obs (GameObservation): Final game state observation.
        """
        from bots.rl_bots.util.reward_system import calculate_rewards_obs
        try:
            # Calculate terminal reward
            terminal_reward = calculate_rewards_obs(final_obs, immediate=False)

            # Assign terminal reward to the last transition
            if self.experience_buffer:
                self.experience_buffer[-1]["reward"] = terminal_reward
        except Exception as e:
            print(f"Error finalizing game: {e}")

    def clear_experience_buffer(self):
        """
        Clear the experience buffer after storing it externally.
        """
        self.experience_buffer.clear()
