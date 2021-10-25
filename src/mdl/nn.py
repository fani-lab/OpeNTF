import torch
from torch import nn
from torch.nn.functional import leaky_relu


class DNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, param['l'][0])
        hl = []
        for i in range(1, len(param['l'])):
            hl.append(nn.Linear(param['l'][i - 1], param['l'][i]))
        self.hidden_layer = nn.ModuleList(hl)
        self.fc2 = nn.Linear(param['l'][-1], output_size)
        self.dp = nn.Dropout(0.5)
        self.initialize_weights()

    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        for i, l in enumerate(self.hidden_layer):
            x = leaky_relu(l(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = torch.clamp(torch.sigmoid(x), min=1.e-6, max=1. - 1.e-6)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)