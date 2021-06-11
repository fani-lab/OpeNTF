import random
import torch
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

fnn = {
    'd': 50,  # size of hidden space
    'lr': 0.1,  # learning rate
    'b': 3,  # batch
    'e': 50000,  # epochs
}

sgns = {
    'd': 50,  # size of hidden space
    'lr': 0.1,  # learning rate
    'b': 3,  # batch
    'e': 50000,  # epochs
    'ns': 2,  # negative samples
}
