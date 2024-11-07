# model.py

import torch.nn as nn

class TrumpPredictorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TrumpPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
