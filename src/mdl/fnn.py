import torch
from torch.nn.functional import leaky_relu, sigmoid
from torch import nn  


class FNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, param['d'])
        self.fc2 = nn.Linear(param['d'], output_size)
        self.dp = nn.Dropout(0.5)

        self.initialize_weights()

    def forward(self, x):
        x = self.dp(leaky_relu(self.fc1(x)))
        x = self.fc2(x)
        x = torch.clamp(torch.sigmoid(x), min=1.e-6, max=1. - 1.e-6)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
