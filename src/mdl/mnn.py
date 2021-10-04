import torch
from torch import nn  
from torch.nn.functional import leaky_relu, sigmoid

class MNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(MNN, self).__init__()
        self.fc1 = nn.Linear(input_size,param['d'])
        self.hidden_layer = nn.ModuleList([nn.Linear(param['d'],param['d']) for i in range(param['l'])])
        self.fc2 = nn.Linear(param['d'],output_size)
        self.dp = nn.Dropout(0.5)
        self.initialize_weights()

    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        for i, l in enumerate(self.hidden_layer):
            x = leaky_relu(l(x))
        x = self.dp(x)
        x = self.fc2(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

