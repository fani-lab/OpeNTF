import os
import time
import numpy as np
import matplotlib as plt
import json
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import param  # do not remove this
from cmn.team import Team
from mdl.bnn import BNN
from mdl.custom_dataset import TFDataset
import fnn_main

#TODO: there is huge code overlapp with bnn training and fnn training. Trying to generalize as we did in test and eval

def learn(splits, indexes, vecs, params, output):
    layers = params['l'];learning_rate = params['lr'];batch_size = params['b'];num_epochs = params['e'];nns = params['nns'];ns = params['ns']
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
        model = BNN(input_size=input_size, output_size=output_size, param=params).to(fnn_main.device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
                    X = X.float().to(device=fnn_main.device)  # Get data to cuda if possible
                    y = y.float().to(device=fnn_main.device)
                    if phase == 'train':
                        model.train(True)  # scheduler.step()
                        # forward
                        optimizer.zero_grad()
                        layer_loss, y_ = model.sample_elbo(X.squeeze(1), y, 1)
                        loss = fnn_main.cross_entropy(y_.to(fnn_main.device), y, ns, nns, unigram) + layer_loss/batch_size
                        # backward
                        loss.backward()
                        # clip_grad_value_(model.parameters(), 1)
                        optimizer.step()
                        train_running_loss += loss.item()
                    else:  # valid
                        model.train(False)  # Set model to valid mode
                        layer_loss, y_ = model.sample_elbo(X.squeeze(1), y, 1)
                        loss = fnn_main.cross_entropy(y_.to(fnn_main.device), y, ns, nns, unigram) + layer_loss/batch_size
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
            torch.save(model.state_dict(), f"{output}/state_dict_model.f{foldidx}.e{epoch}.pt", pickle_protocol=4)
            scheduler.step(valid_running_loss / X_valid.shape[0])

        model_path = f"{output}/state_dict_model_f{foldidx}.pt"

        # Save
        torch.save(model.state_dict(), model_path, pickle_protocol=4)
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

def main(splits, vecs, indexes, output, settings, cmd):
    output = f"{output}/t{vecs['skill'].shape[0]}.s{vecs['skill'].shape[1]}.m{vecs['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings.items() if v])}"
    if not os.path.isdir(output): os.makedirs(output)

    if 'train' in cmd: learn(splits, indexes, vecs, settings, output)
    if 'test' in cmd: fnn_main.test(BNN, output, splits, indexes, vecs, settings, on_train_valid_set=False)
    if 'eval' in cmd: fnn_main.evaluate(output, splits, vecs, on_train_valid_set=False, per_instance=False)
    if 'plot' in cmd: fnn_main.plot_roc(output, splits, on_train_valid_set=False)
