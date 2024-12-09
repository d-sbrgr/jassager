import torch
import torch.nn as nn
import torch.nn.functional as F


class JassNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(JassNet, self).__init__()
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Policy head
        self.policy_head = nn.Linear(256, action_dim)

        # Value head
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layers
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value
