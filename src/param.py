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
            'dnn': {
                'd': 100,    # size of hidden space
                'lr': 0.1,  # learning rate
                'b': 1000,     # batch_size
                'e': 2,     # epochs
            },
            'sgns':{
                'd': 100,
                'lr': 0.01,
                'b': 3,
                'e': 10,
                'ns': 3,
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
            'min_nteam': 2,
            'min_team_size': 2,
        },
        'ncore': 4,
        'bucket_size': 1000
    },
}