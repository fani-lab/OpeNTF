import random
random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import numpy as np
np.random.seed(0)

fnn = {
    'd': 50,#size of hidden space
    'lr': 0.03,#learning rate
    'b': 1, #batch
    'e': 25000, #epochs
}

sgns = {
    'd': 50,#size of hidden space
    'lr': 0.03,#learning rate
    'b': 1, #batch
    'e': 25000, #epochs
    'ns' : 2, #negative samples
}