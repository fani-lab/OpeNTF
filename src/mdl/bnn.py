import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from .bayesian_layer import bayesianLayer

class BNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super().__init__()
        self.h1 = bayesianLayer(input_size, param['l'][0])
        self.h2 = bayesianLayer(param['l'][0], param['l'][0])
        self.out = bayesianLayer(param['l'][0], output_size)
        self.output_size = output_size

    def forward(self, x):
        x = leaky_relu(self.h1(x))
        x = leaky_relu(self.h2(x))
        x = torch.clamp(torch.sigmoid(self.out(x)), min=1.e-6, max=1. - 1.e-6)
        return x

    def log_prior(self):
        return self.h1.log_prior + self.h2.log_prior + self.out.log_prior

    def log_post(self):
        return self.h1.log_post + self.h2.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        outputs = torch.zeros(target.shape[0], samples, self.output_size)
        # print(outputs[0].size())
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        #log_likes = torch.zeros(samples)
        for i in range(samples):
          outputs[:, i, :] = self(input)
          log_priors[i] = self.log_prior()
          log_posts[i] = self.log_post()
          #log_likes[i] = torch.log(outputs[i, torch.arange(outputs.shape[1]), target]).sum(dim=-1)
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        loss = log_post - log_prior
        #log_likes = F.nll_loss(outputs.mean(0), target, size_average=False)
        # log_likes = F.nll_loss(outputs.mean(0), target, reduction='sum')
        # loss = (log_post - log_prior)/num_batches + log_likes
        return loss, outputs