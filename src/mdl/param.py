import random
import torch
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

fnn = {
    'd': 10,  # size of hidden space
    'lr': 0.001,  # learning rate
    'b': 4,  # batch
    'e': 2,  # epochs
}

sgns = {
    'd': 100,  # size of hidden space
    'lr': 0.01,  # learning rate
    'b': 3,  # batch
    'e': 10,  # epochs
    'ns': 3,  # negative samples
}
