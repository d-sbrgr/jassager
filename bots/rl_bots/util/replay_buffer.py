import random

# Debug flag
debug = False

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # Tracks the circular buffer position
        self.winning_games_count = 0
        self.losing_games_count = 0

    def push(self, transition):
        """Store a transition in the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        if debug:
            print(f"Debug replay_buffer.py Buffer size: {len(self.buffer)}, Position: {self.position}")

    def push_with_prio(self, transition, is_winning_game):
        """
        Store a transition in the buffer, prioritizing winning games and ensuring
        losing games do not exceed winning games.

        Args:
            transition (tuple): The transition to store (state, action, reward, next_state, done).
            is_winning_game (bool): True if the transition is part of a winning game, False otherwise.
        """
        if is_winning_game:
            # Always store winning game transitions
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity
            self.winning_games_count += 1
        else:
            # Only store losing game transitions if the count allows
            if self.losing_games_count < self.winning_games_count:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                self.buffer[self.position] = transition
                self.position = (self.position + 1) % self.capacity
                self.losing_games_count += 1

        if debug:
            print(f"Debug ReplayBuffer: Buffer size: {len(self.buffer)}, Position: {self.position}, "
                  f"Winning games: {self.winning_games_count}, Losing games: {self.losing_games_count}")

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            print("Warning: Not enough samples in buffer.")
            return []
        return random.sample(self.buffer, batch_size)

    def sample_recent(self, batch_size):
        # Calculate the most recent half of the buffer, or the full buffer if it's too small
        recent_size = max(len(self.buffer) // 2, batch_size)
        recent_samples = self.buffer[-recent_size:]

        # Sample the minimum between batch_size and the actual number of recent samples
        return random.sample(recent_samples, min(len(recent_samples), batch_size))

    def __len__(self):
        return len(self.buffer)
