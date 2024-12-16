import random

# Debug flag
debug = True

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # Tracks the circular buffer position

    def push(self, transition):
        """Store a transition in the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        if debug:
            print(f"Debug replay_buffer.py Buffer size: {len(self.buffer)}, Position: {self.position}")

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            print("Warning: Not enough samples in buffer.")
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
