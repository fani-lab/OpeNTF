import random
import torch
import numpy as np

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

np.random.seed(0)

settings = {
    "gpus": "2",
    "model": {
        "baseline": {
            "random": {"b": 128},
            "fnn": {
                "l": [128],  # list of number of nodes in each layer
                "lr": 0.1,  # learning rate
                "b": 128,  # batch size
                "e": 1,  # epoch
                "nns": 3,  # number of negative samples
                "ns": "unigram_b",  # 'none', 'uniform', 'unigram', 'unigram_b'
                "loss": "SL",  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
            },
            "bnn": {
                "l": [128],  # list of number of nodes in each layer
                "lr": 0.1,  # learning rate
                "b": 128,  # batch size
                "e": 5,  # epoch
                "nns": 3,  # number of negative samples
                "ns": "unigram_b",  # 'uniform', 'unigram', 'unigram_b'
                "s": 1,  # # sample_elbo for bnn
                "loss": "SL",  # 'SL'-> superloss, 'DP' -> Data Parameters, 'normal' -> Binary Cross Entropy
            },
            "caser": {},
            "rrn": {"with_zero": True},
            "emb": {
                "d": 128,  # embedding dimension
                "e": 100,  # epoch
                "dm": 1,  # training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
                "w": 1,  # cooccurrence window
            },
        },
        "cmd": [
            # "train",
            # "test",
            # "eval",
            # "fair",
        ],  # 'train', 'test', 'eval', 'plot', 'agg', 'fair'
        "nfolds": 5,
        "train_test_split": 0.85,
        "step_ahead": 2,  # for now, it means that whatever are in the last [step_ahead] time interval will be the test set!
    },
    "data": {
        "domain": {
            "dblp": {},
            "uspt": {},
            "imdb": {},
            "gith": {},
        },
        "location_type": "country",  # should be one of 'city', 'state', 'country' and represents the location of members in teams (not the location of teams)

        # filters that can be applied in first pass of preprocessing
        "pass1_filters": {
            "remove_duplicates": False, # True for removing duplicates
            "remove_empty_skills": False, # True for removing empty skills
            "remove_empty_members": False, # True for removing empty teams
            "min_team_size": -1, # -1 for no filtering
            "max_team_size": -1, # -1 for no filtering
            "min_skills": -1, # -1 for no filtering
            "max_skills": -1, # -1 for no filtering
        },
        "filter": {
            # filters included in output folder name
        },
        # filters that require new rounds of data processing
        "passn_filters": {
            "min_nteam": -1, # -1 for no filtering
        },
        "processing": {
            "debug_logs": False, # True for debug logs
            "raw_logs": False, # True for raw logs
            "nthreads": 0,  # <= 0 for all cores
            "cpu_batch_size": 25_000, # -1 for domain-specific batch size in their own code in src/cmn/<domain>.py
            "gpu_batch_size": 500_000, # -1 for domain-specific batch size in their own code in src/cmn/<domain>.py
            "make_toy_data": True, # True for making toy data
            "toy_data_size": 100, # Number of teams to use for toy data
        },
    },
    "fair": {
        "np_ratio": None,
        "fairness": [
            "det_greedy",
        ],
        "k_max": None,
        "fairness_metrics": {"ndkl"},
        "utility_metrics": {"map_cut_2,5,10"},
        "eq_op": False,
        "mode": 0,
        "core": -1,
        "attribute": ["gender", "popularity"],
    }
}
