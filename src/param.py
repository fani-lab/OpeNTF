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
            'random': {
                'b': 128
            },
            'fnn':{
                'l': [100],  # list of number of nodes in each layer
                'lr': 0.01, # learning rate 0.001
                'b': 128,  # batch size used to  be 4
                'e': 20,  # epoch used to be 10
                'nns': 3,  # number of negative samples
                'ns': 'female_bias',  # 'none', 'uniform', 'unigram', 'unigram_b', 'female_bias', 'all-female_bias
                'loss': 'normal',  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
            },
            'bnn':{
                'l': [128],  # list of number of nodes in each layer
                'lr': 0.1,  # learning rate
                'b': 128,  # batch size
                'e': 5,  # epoch
                'nns': 3,  # number of negative samples
                'ns': 'female_bias',  # 'uniform', 'unigram', 'unigram_b'
                's': 1,  # # sample_elbo for bnn
                'loss': 'normal',  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
            },
            'nmt': {
                'base_config': './mdl/nmt_config.yaml'
            },
            'caser': {},
            'rrn': {
                'with_zero': True
            },
            'emb':{
                'd': 100,# embedding dimension
                'e': 100,# epoch
                'dm': 1,# training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
                'w': 1 #cooccurrence window
            }
        },
        'cmd': ['train','test','eval'],  # 'train', 'test', 'eval', 'plot', 'agg', 'fair'
       #' nfolds': 3, toy
         'nfolds': 5,
        'train_test_split': 0.85,
        'step_ahead': 1,#for now, it means that whatever are in the last [step_ahead] time interval will be the test set!
    },
    'data':{
        'domain': {
            'dblp':{},
            'uspt':{},
            'imdb':{},
        },
        'location_type': 'country', #should be one of 'city', 'state', 'country' and represents the location of members in teams (not the location of teams)
        'filter': {
            'min_nteam': 75,
            'min_team_size': 3,
        },
        'parallel': 1,
        'ncore': 4,# <= 0 for all
        'bucket_size': 500
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
