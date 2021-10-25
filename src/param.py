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
                'l': [100],  # list of number of nodes in each layer
                'lr': 0.1,  # learning rate
                'b': 4096,  # batch size
                'e': 2,  # epoch
                'ns': 5,  # number of negative samples
                's': 'none',  # 'none', 'uniform', 'unigram', 'unigram_b'
            },
        },
        'cmd': ['train', 'plot', 'eval', 'test'],  # 'train', 'plot', 'eval', 'test'
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