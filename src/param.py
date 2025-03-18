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
        "filters": {
            "common": {
                # None/False/0 means no filtering
                "remove_dup_teams": False,
                "remove_empty_skills_teams": False,
                "remove_empty_experts_teams": False,
                "min_team_size": None,
                "max_team_size": None,
                "min_skills": None,
                "max_skills": None,
                "min_teams_per_expert": None,
                "max_teams_per_expert": None,
            },
            # Domain-specific filters
            # Each has their own min/max years because we may want different ranges for different domains
            "gith": {
                # repository creation year
                "min_year": None,  # None means no filtering
                "max_year": None,  # None means no filtering
            },
            "dblp": {
                # paper publication year
                "min_year": None,  # None means no filtering
                "max_year": None,  # None means no filtering
            },
            "uspt": {
                # Add USPT-specific filters here as needed
            },
            "imdb": {
                # Add IMDB-specific filters here as needed
            },
        },
        "processing": {
            "debug_logs": False,  # True for debug logs
            "raw_logs": False,  # True for raw logs
            "nthreads": 0.75,  # refer to src/cmn_v3/helper_functions/get_nthreads.py for detailed info on possible values
            "cpu_batch_size": 25_000,  # -1 for domain-specific batch size in their own code in src/cmn/<domain>.py
            "gpu_batch_size": 500_000,  # -1 for domain-specific batch size in their own code in src/cmn/<domain>.py
            "make_toy_data": True,  # True for making toy data
            "toy_data_size": 100,  # Number of teams to use for toy data
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
    },
}
