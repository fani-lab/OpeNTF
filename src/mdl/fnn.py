import torch
from torch.nn.functional import relu
from torch import nn  

class FNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, param['d'])
        self.fc2 = nn.Linear(param['d'], output_size)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x