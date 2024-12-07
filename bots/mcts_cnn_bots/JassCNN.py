import torch
import torch.nn as nn


class JassCNN(nn.Module):
    def __init__(self, input_channels=19, num_actions=36):
        super(JassCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc_shared = nn.Linear(256 * 4 * 9, 256)
        self.dropout = nn.Dropout(0.5)

        self.fc_policy = nn.Linear(256, num_actions)
        self.fc_value1 = nn.Linear(256, 64)
        self.fc_value2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)

        shared = torch.relu(self.fc_shared(x))
        shared = self.dropout(shared)

        policy = torch.softmax(self.fc_policy(shared), dim=-1)
        value = torch.relu(self.fc_value1(shared))
        value = torch.sigmoid(self.fc_value2(value))

        return policy, value