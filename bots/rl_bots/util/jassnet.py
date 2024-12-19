import torch.nn as nn
import torch.nn.functional as F

class JassNet(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(JassNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Larger hidden layer
        self.fc2 = nn.Linear(512, 256)
        self.gru = nn.GRU(256, 128, batch_first=True)  # Track sequences
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, _ = self.gru(x.unsqueeze(0))  # Add sequential context
        x = x.squeeze(0)
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        return policy, value


