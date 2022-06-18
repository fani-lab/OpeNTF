import os, time, json, re
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.functional import leaky_relu
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from mdl.ntf import Ntf
from mdl.cds import TFDataset

from cmn.team import Team

class Fnn(Ntf):
    def __init__(self):
        super(Fnn, self).__init__()

    def init(self, input_size, output_size, param):
        self.fc1 = nn.Linear(input_size, param['l'][0])
        hl = []
        for i in range(1, len(param['l'])):
            hl.append(nn.Linear(param['l'][i - 1], param['l'][i]))
        self.hidden_layer = nn.ModuleList(hl)
        self.fc2 = nn.Linear(param['l'][-1], output_size)
        self.initialize_weights()
        return self

    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        for i, l in enumerate(self.hidden_layer):
            x = leaky_relu(l(x))
        x = self.fc2(x)
        x = torch.clamp(torch.sigmoid(x), min=1.e-6, max=1. - 1.e-6)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def cross_entropy(self, y_, y, ns, nns, unigram):
        if ns == "uniform": return self.ns_uniform(y_, y, nns)
        if ns == "unigram": return self.ns_unigram(y_, y, unigram, nns)
        if ns == "unigram_b": return self.ns_unigram_mini_batch(y_, y, nns)
        return self.weighted(y_, y)

    def weighted(self, logits, targets, pos_weight=2.5):
        targets = targets.squeeze(1)
        logits = logits.squeeze(1)
        return (-targets * torch.log(logits) * pos_weight + (1 - targets) * - torch.log(1 - logits)).sum()

    def ns_uniform(self, logits, targets, neg_samples=5):
        targets = targets.squeeze(1)
        logits = logits.squeeze(1)
        random_samples = torch.zeros_like(targets)
        for b in range(targets.shape[0]):
            k_neg_idx = torch.randint(0, targets.shape[1], (neg_samples,))
            cor_idx = torch.nonzero(targets[b].cpu(), as_tuple=True)[0]
            for idx in k_neg_idx:
                if idx not in cor_idx:
                    random_samples[b][idx] = 1
        return (-targets * torch.log(logits) - random_samples * torch.log(1 - logits)).sum()

    def ns_unigram(self, logits, targets, unigram, neg_samples=5):
        targets = targets.squeeze(1)
        logits = logits.squeeze(1)
        random_samples = torch.zeros_like(targets)
        for b in range(targets.shape[0]):
            rand = np.random.rand(targets.shape[1])
            neg_rands = (rand > unigram) * 1
            neg_idx = torch.nonzero(torch.tensor(neg_rands), as_tuple=True)[1]
            k_neg_idx = np.random.choice(neg_idx, neg_samples)
            cor_idx = torch.nonzero(targets[b], as_tuple=True)[0]
            for idx in k_neg_idx:
                if idx not in cor_idx:
                    random_samples[b][idx] = 1
        return (-targets * torch.log(logits) - random_samples * torch.log(1 - logits)).sum()

    def ns_unigram_mini_batch(self, logits, targets, neg_samples=5):
        targets = targets.squeeze(1)
        logits = logits.squeeze(1)
        random_samples = torch.zeros_like(targets)
        n_paper_per_author = torch.sum(targets, dim=0) + 1
        unigram = (n_paper_per_author / (targets.shape[0] + targets.shape[1])).cpu()

        for b in range(targets.shape[0]):
            rand = torch.rand(targets.shape[1])
            neg_rands = (rand > unigram) * 1
            neg_idx = torch.nonzero(torch.tensor(neg_rands), as_tuple=True)[0]
            k_neg_idx = np.random.choice(neg_idx, neg_samples)
            cor_idx = torch.nonzero(targets[b], as_tuple=True)[0]
            for idx in k_neg_idx:
                if idx not in cor_idx:
                    random_samples[b][idx] = 1
        return (-targets * torch.log(logits) - random_samples * torch.log(1 - logits)).sum()

    def learn(self, splits, indexes, vecs, params, prev_model, output):

        layers = params['l'];
        learning_rate = params['lr'];
        batch_size = params['b'];
        num_epochs = params['e'];
        nns = params['nns'];
        ns = params['ns']
        # input_size = len(indexes['i2s'])
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
            X_train = vecs['skill'][splits['folds'][foldidx]['train'], :];
            y_train = vecs['member'][splits['folds'][foldidx]['train']]
            X_valid = vecs['skill'][splits['folds'][foldidx]['valid'], :];
            y_valid = vecs['member'][splits['folds'][foldidx]['valid']]

            # Building custom dataset
            training_matrix = TFDataset(X_train, y_train)
            validation_matrix = TFDataset(X_valid, y_valid)

            # Generating data loaders
            training_dataloader = DataLoader(training_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
            validation_dataloader = DataLoader(validation_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
            data_loaders = {"train": training_dataloader, "valid": validation_dataloader}

            # Initialize network
            self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)
            if prev_model: self.load_state_dict(torch.load(prev_model[foldidx]))

            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
            # scheduler = StepLR(optimizer, step_size=3, gamma=0.9)

            train_loss_values = []
            valid_loss_values = []
            fold_time = time.time()
            # Train Network
            for epoch in range(num_epochs):
                train_running_loss = valid_running_loss = 0.0
                # Each epoch has a training and validation phase
                for phase in ['train', 'valid']:
                    for batch_idx, (X, y) in enumerate(data_loaders[phase]):
                        torch.cuda.empty_cache()
                        X = X.float().to(device=self.device)  # Get data to cuda if possible
                        y = y.float().to(device=self.device)
                        if phase == 'train':
                            self.train(True)  # scheduler.step()
                            # forward
                            optimizer.zero_grad()
                            y_ = self.forward(X)
                            loss = self.cross_entropy(y_, y, ns, nns, unigram)
                            # backward
                            loss.backward()
                            # clip_grad_value_(model.parameters(), 1)
                            optimizer.step()
                            train_running_loss += loss.item()
                        else:  # valid
                            self.train(False)  # Set model to valid mode
                            y_ = self.forward(X)
                            loss = self.cross_entropy(y_, y, ns, nns, unigram)
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
                torch.save(self.state_dict(), f"{output}/state_dict_model.f{foldidx}.e{epoch}.pt", pickle_protocol=4)
                scheduler.step(valid_running_loss / X_valid.shape[0])

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
                plt.show()

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False):
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
            if per_epoch: modelfiles += [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match(f'state_dict_model.f{foldidx}.e\d+.pt', _)]

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
                    with torch.no_grad():
                        y_pred = torch.empty(0, dl.dataset.output.shape[1])
                        for x, y in dl:
                            x = x.to(device=self.device)
                            scores = self.forward(x)
                            scores = scores.squeeze(1).cpu().numpy()
                            y_pred = np.vstack((y_pred, scores))
                    epoch = modelfile.split('.')[-2] + '.' if per_epoch else ''
                    epoch = epoch.replace(f'f{foldidx}.', '')
                    torch.save(y_pred, f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)

