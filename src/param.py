import random
import torch
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

settings = {
    'model':{
        'baseline': {
            'nn':{
                'd': 100,  # number of nodes for each layer
                'l': 1,  # number of layers
                'lr': 0.1,  # learning rate
                'b': 4096,  # batch size
                'e': 2,  # epoch
                'ns': 5,  # number of negative samples
                's': 'none', # 'none', 'uniform', 'unigram', 'unigram_b'
            },
        },
        'cmd': ['train', 'plot', 'test', 'eval'],
        'splits': 5
    },
    'data':{
        'domain': {
            'dblp':{},
            'uspt':{},
            'imdb':{},
        },
        'filter': {
            'min_nteam': 30,
            'min_team_size': 3,
        },
        'ncore': 5,
        'bucket_size': 500
    },
}