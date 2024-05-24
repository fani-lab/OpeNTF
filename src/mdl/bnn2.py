# https://github.com/IntelLabs/bayesian-torch

import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

import os
import time
import matplotlib as plt
import json
import matplotlib.pyplot as plt
import numpy as np
import re

from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
from torch.distributions import Normal

# We benefit from Josh Feldman's great blog at https://joshfeldman.net/WeightUncertainty/

from mdl.cds import TFDataset
from mdl.fnn import Fnn
from cmn.team import Team
from cmn.tools import merge_teams_by_skills

from cmn.tools import get_class_data_params_n_optimizer, adjust_learning_rate, apply_weight_decay_data_parameters
from mdl.cds import SuperlossDataset
from mdl.earlystopping import EarlyStopping
from mdl.superloss import SuperLoss

class Bnn(nn.Module):

    def __init__(self):
        super(Bnn, self).__init__()

    def init(self, input_size, output_size, param):
        self.h1 = nn.Linear(input_size, param['l'][0])
        hl = []
        for i in range(1, len(param['l'])):
            hl.append(nn.Linear(param['l'][i - 1], param['l'][i]))
        self.hidden_layer = nn.ModuleList(hl)
        self.out = nn.Linear(param['l'][-1], output_size)
        self.initialize_weights()
        const_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }
        self.output_size = output_size
        dnn_to_bnn(self, const_bnn_prior_parameters)
        return self




if __name__ == '__main__':

    from param import settings

    param = settings['model']['bnn']

    # load teamsvecs

    bnn = Bnn()
    bnn.init(input_size, output_size, param)