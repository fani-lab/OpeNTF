import os
import time
import matplotlib as plt
import json
import matplotlib.pyplot as plt
import numpy as np
import re

import torch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
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


class Bnn(Fnn):
    def __init__(self):
        super().__init__()

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
            "type": "Flipout",  # Flipout or Reparameterization
            "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
            "moped_delta": 0.5,
        }
        self.output_size = output_size
        dnn_to_bnn(self, const_bnn_prior_parameters)
        return self

    def forward(self, x):
        x = leaky_relu(self.h1(x))
        for i, l in enumerate(self.hidden_layer):
            x = leaky_relu(l(x))
        x = torch.clamp(torch.sigmoid(self.out(x)), min=1.e-6, max=1. - 1.e-6)
        return x

    def log_prior(self):
        return self.h1.log_prior + torch.sum(
            torch.as_tensor([hl.log_prior for hl in self.hidden_layer])) + self.out.log_prior

    def log_post(self):
        return self.h1.log_post + torch.sum(
            torch.as_tensor([hl.log_post for hl in self.hidden_layer])) + self.out.log_post

    def sample_elbo(self, input, target, samples):
        outputs = torch.zeros(target.shape[0], samples, self.output_size)
        # print(outputs[0].size())
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        # log_likes = torch.zeros(samples)
        for i in range(samples):
            outputs[:, i, :] = self(input)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            # log_likes[i] = torch.log(outputs[i, torch.arange(outputs.shape[1]), target]).sum(dim=-1)
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        loss = log_post - log_prior
        outputs = outputs.mean(axis=1)
        # log_likes = F.nll_loss(outputs.mean(0), target, size_average=False)
        # log_likes = F.nll_loss(outputs.mean(0), target, reduction='sum')
        # loss = (log_post - log_prior)/num_batches + log_likes
        return loss.to(self.device), outputs.to(self.device)

    # TODO: there is huge code overlapp with bnn training and fnn training. Trying to generalize as we did in test and eval
    def learn(self, splits, indexes, vecs, params, prev_model, output):
        loss_type = params['loss']

        learning_rate = params['lr']
        batch_size = params['b']
        num_epochs = params['e']
        nns = params['nns']
        ns = params['ns']
        s = params['s']
        weight = params['weight']
        input_size = vecs['skill'].shape[1]
        output_size = len(indexes['i2c'])

        unigram = Team.get_unigram(vecs['member'])

        # Prime a dict for train and valid loss
        train_valid_loss = dict()
        for i in range(len(splits['folds'].keys())):
            train_valid_loss[i] = {'train': [], 'valid': []}

        start_time = time.time()
        # Training K-fold
        for foldidx in splits['folds'].keys():
            # Retrieving the folds
            X_train = vecs['skill'][splits['folds'][foldidx]['train'], :]
            y_train = vecs['member'][splits['folds'][foldidx]['train']]
            X_valid = vecs['skill'][splits['folds'][foldidx]['valid'], :]
            y_valid = vecs['member'][splits['folds'][foldidx]['valid']]

            # Building custom dataset
            training_matrix = SuperlossDataset(X_train, y_train)
            validation_matrix = SuperlossDataset(X_valid, y_valid)

            # Generating data loaders
            training_dataloader = DataLoader(training_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
            validation_dataloader = DataLoader(validation_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
            data_loaders = {"train": training_dataloader, "valid": validation_dataloader}

            # Initialize network
            self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)
            if prev_model: self.load_state_dict(torch.load(prev_model[foldidx]))

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), 0.001)
            # scheduler = StepLR(optimizer, step_size=3, gamma=0.9)

            train_loss_values = []
            valid_loss_values = []
            fold_time = time.time()
            # Train Network

            earlystopping = EarlyStopping(patience=5, verbose=False, delta=0.01, path=f"{output}/state_dict_model.f{foldidx}.pt", trace_func=print)
            for epoch in range(num_epochs):
                train_running_loss = valid_running_loss = 0.0
                # Each epoch has a training and validation phase
                for phase in ['train', 'valid']:
                    for batch_idx, (X, y, index) in enumerate(data_loaders[phase]):
                        torch.cuda.empty_cache()
                        X = X.float().to(device=self.device)  # Get data to cuda if possible
                        y = y.float().to(device=self.device)
                        if phase == 'train':
                            self.train(True)  # scheduler.step()
                            # forward
                            optimizer.zero_grad()
                            y_ = self(X)
                            kl = get_kl_loss(self)
                            ce_loss = self.cross_entropy(y_, y, ns, nns, unigram, weight)
                            loss = ce_loss + kl / params['b'] # is the batch size equal everytime?

                            loss.backward()
                            optimizer.step()
                            train_running_loss += loss.item()
                        else:  # valid
                            self.train(False)  # Set model to valid mode
                            y_ = self(X)
                            kl = get_kl_loss(self)
                            ce_loss = self.cross_entropy(y_, y, ns, nns, unigram, weight)
                            loss = ce_loss + kl / params['b']
                            valid_running_loss += loss.item()
                        print(
                            f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {epoch}/{num_epochs - 1}, Minibatch {batch_idx}/{int(X_train.shape[0] / batch_size)}, Phase {phase}'
                            f', Running Loss {phase} {loss.item()}'
                            f", Time {time.time() - fold_time}, Overall {time.time() - start_time} "
                        )
                    # Appending the loss of each epoch to plot later
                    if phase == 'train':
                        train_loss_values.append(train_running_loss / X_train.shape[0])
                    else:
                        valid_loss_values.append(valid_running_loss / X_valid.shape[0])
                    print(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {epoch}/{num_epochs - 1}'
                          f', Running Loss {phase} {train_loss_values[-1] if phase == "train" else valid_loss_values[-1]}'
                          f", Time {time.time() - fold_time}, Overall {time.time() - start_time} "
                          )
                # torch.save(self.state_dict(), f"{output}/state_dict_model.f{foldidx}.e{epoch}.pt", pickle_protocol=4)
                earlystopping(valid_loss_values[-1], self)
                if earlystopping.early_stop:
                    print(f"Early Stopping Triggered at epoch: {epoch}")
                    break

            model_path = f"{output}/state_dict_model.f{foldidx}.pt"

            # Save
            torch.save(self.state_dict(), model_path, pickle_protocol=4)
            train_valid_loss[foldidx]['train'] = train_loss_values
            train_valid_loss[foldidx]['valid'] = valid_loss_values

        print(f"It took {time.time() - start_time} to train the model.")
        with open(f"{output}/train_valid_loss.json", 'w') as outfile:
            json.dump(train_valid_loss, outfile)
            for foldidx in train_valid_loss.keys():
                plt.figure()
                plt.plot(train_valid_loss[foldidx]['train'], label='Training Loss')
                plt.plot(train_valid_loss[foldidx]['valid'], label='Validation Loss')
                plt.legend(loc='upper right')
                plt.title(f'Training and Validation Loss for fold #{foldidx}')
                plt.savefig(f'{output}/f{foldidx}.train_valid_loss.png', dpi=100, bbox_inches='tight')
                # plt.show() # temporarily turning this off, to better automate in bash environment

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False,
             merge_skills=False):

        if not os.path.isdir(model_path): raise Exception("The model does not exist!")
        # input_size = len(indexes['i2s'])
        input_size = vecs['skill'].shape[1]
        output_size = len(indexes['i2c'])

        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = TFDataset(X_test, y_test)
        test_dl = DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

        for foldidx in splits['folds'].keys():
            modelfiles = [f'{model_path}/state_dict_model.f{foldidx}.pt']
            if per_epoch: modelfiles += [f'{model_path}/{_}' for _ in os.listdir(model_path) if
                                         re.match(f'state_dict_model.f{foldidx}.e\d+.pt', _)]

            for modelfile in modelfiles:
                self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)
                self.load_state_dict(torch.load(modelfile))
                self.eval()

                for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
                    if pred_set != 'test':
                        X = vecs['skill'][splits['folds'][foldidx][pred_set], :]
                        y = vecs['member'][splits['folds'][foldidx][pred_set]]
                        matrix = TFDataset(X, y)
                        dl = DataLoader(matrix, batch_size=params['b'], shuffle=True, num_workers=0)
                    else:
                        X = X_test; y = y_test; matrix = test_matrix
                        dl = test_dl

                    torch.cuda.empty_cache()
                    ####
                    # criterion = torch.nn.CrossEntropyLoss()
                    # criterion = self.cross_entropy(self, y_, y, ns, nns, unigram, weight)
                    y_pred = []
                    test_running_loss = 0.0

                    with torch.no_grad():
                        for batch_idx, (X, y) in enumerate(dl):
                            X = X.float().to(self.device)
                            # y = y.float().to(self.device)  # Ensure y_batch is long for CrossEntropyLoss

                            y_ = self(X)
                            # kl = get_kl_loss(self)
                            # ce_loss = criterion(y_, y)
                            # loss = ce_loss + kl / params['b']
                            # test_running_loss += loss.item()

                            y_pred_batch = y_.cpu().numpy()
                            y_pred.append(y_pred_batch)

                            # print(f'Fold {foldidx}, Batch {batch_idx}, {pred_set} set, Loss: {loss.item()}')
                    y_pred = np.vstack(y_pred).squeeze()

                    epoch = modelfile.split('.')[-2] + '.' if per_epoch else ''
                    epoch = epoch.replace(f'f{foldidx}.', '')
                    torch.save(y_pred, f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)