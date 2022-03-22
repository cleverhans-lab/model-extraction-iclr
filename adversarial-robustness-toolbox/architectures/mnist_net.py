import torch
from torch import nn as nn
from torch.nn import functional as F


class MnistNet(nn.Module):
    def __init__(self, name, args, input_channels=1, input_size=28):
        super(MnistNet, self).__init__()
        self.name = name
        self.args = args
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        if input_size == 28:
            n = 9216
        elif input_size == 32:
            n = 12544
        else:
            raise Exception(f'Unsupported input_size: {input_size}.')
        self.fc1 = nn.Linear(n, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
