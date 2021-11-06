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
            'fnn':{
                'l': [100],  # list of number of nodes in each layer
                'lr': 0.1,  # learning rate
                'b': 4096,  # batch size
                'e': 2,  # epoch
                'nns': None,  # number of negative samples
                'ns': None,  # 'uniform', 'unigram', 'unigram_b'
            },
        },
        'cmd': ['eval'],  # 'train', 'test', 'eval'
        'nfolds': 5,
        'train_test_split': 0.85
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