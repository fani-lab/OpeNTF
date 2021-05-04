import torch
from torch.nn.functional import relu
from torch import nn  



class FNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x