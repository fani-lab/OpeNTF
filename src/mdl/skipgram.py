from torch import nn  

class SGNS(nn.Module):
    def __init__(self, input_size, output_size, param):
        super(SGNS, self).__init__()
        self.fc1 = nn.Linear(input_size, param['d'])
        self.fc2 = nn.Linear(param['d'], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x