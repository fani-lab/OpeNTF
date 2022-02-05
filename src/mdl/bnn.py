#We benefit from Josh Feldman's great blog at https://joshfeldman.net/WeightUncertainty/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
import torch.optim as optim
from torch.distributions import Normal

class BNN(nn.Module):
    def __init__(self, input_size, output_size, param):
        super().__init__()
        self.h1 = BayesianLayer(input_size, param['l'][0])
        self.h2 = BayesianLayer(param['l'][0], param['l'][0])
        self.out = BayesianLayer(param['l'][0], output_size)
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

class BayesianLayer(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(self.w_mu.device)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(self.b_mu.device)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w).to(self.w.device)
        b_log_prior = self.prior.log_prob(self.b).to(self.b.device)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)