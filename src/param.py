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
                'b': 4096,     # batch_size
                'e': 2,     # epochs
            },
            'sgns':{
                'd': 100,
                'lr': 0.1,
                'b': 4096,
                'e': 2,
                'ns': 5,
                's': 'unigram'  # 'uniform', 'unigram', 'unigram_b'
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