import torch
import torch.nn as nn


class JassCNN(nn.Module):
    def __init__(self, input_channels=19, num_actions=36):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=2, padding=0)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(2, 2), stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):   
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = nn.functional.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = nn.functional.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01)

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = torch.sigmoid(self.fc_value(x))
        
        return value