import random
import torch
import numpy as np

random.seed(0)
settings = {
    'model':{
        'baseline': {
            'random': {
                'b': 128
            },
            'fnn':{
                'b': 2048,  # batch size
                'e': 25,  # epoch
                'l': [128],  # list of number of nodes in each layer
                'loss': 'normal', # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy 'pos-ce' -> positive ce, 'weighted' -> weighted ce
                'lr': 0.0001,  # learning rate
                'nns': 3,  # number of negative samples
                'ns': 'uniform',  # 'none', 'uniform', 'unigram', 'unigram_b'
                'weight': 2.5, # weight if ns == 'weighted'
            },
            'bnn_old':{
                'l': [128],  # list of number of nodes in each layer
                'lr': 0.01,  # learning rate
                'b': 2048,  # batch size
                'e': 25,  # epoch
                'nns': 3,  # number of negative samples
                'ns': 'uniform',  # 'uniform', 'unigram', 'unigram_b'
                'weight': 5, # weight if ns == 'weighted'
                's': 1,  # # sample_elbo for bnn
                'loss': 'normal',  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
            },
            'bnn':{
                'b': 2048,  # batch size
                'e': 25,  # epoch
                'l': [128],  # list of number of nodes in each layer
                'loss': 'normal',  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
                'lr': 0.01,  # learning rate
                'nns': 3,  # number of negative samples
                'ns': 'uniform',  # 'uniform', 'unigram', 'unigram_b'
                's': 1,  # # sample_elbo for bnn
                'weight': 2.5, # weight if ns == 'weighted'
            },
            'nmt': {
                'base_config': './mdl/nmt_config.yaml'
            },
            'caser': {},
            'rrn': {
                'with_zero': True
            },
            'emb':{
                'e': 100,# max epoch
                'd': 128,# embedding dimension
                'dm': 1,# training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
                'w': 1, #cooccurrence window
                'b' : 128, # 0 means no batching
                'ns' : 2,
            }
        },

        'cmd':['train', 'test', 'eval'],
        # 'cmd': ['eval'],  # 'train', 'test', 'eval', 'plot', 'agg', 'fair'

        'nfolds': 3,
        'train_test_split': 0.85,
        'step_ahead': 2,#for now, it means that whatever are in the last [step_ahead] time interval will be the test set!
    },
    'data':{
        'domain': {
            'dblp':{},
            'uspt':{},
            'imdb':{},
        },
        'location_type': 'country', #should be one of 'city', 'state', 'country' and represents the location of members in teams (not the location of teams)
        'filter': {
            'min_nteam': 5,
            'min_team_size': 2,
        },
        'parallel': 1,
        'ncore': 0,# <= 0 for all
        'bucket_size': 1000
    },
    'fair': {'np_ratio': None,
              'fairness': ['det_greedy',],
              'k_max': None,
              'fairness_metrics': {'ndkl'},
              'utility_metrics': {'map_cut_2,5,10'},
              'eq_op': False,
              'mode': 0,
              'core': -1,
              'attribute': ['gender', 'popularity']},
}
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

