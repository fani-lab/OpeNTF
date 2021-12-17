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
import pandas as pd
import pickle

import param #do not remove this
from cmn.team import Team
from mdl.fnn import FNN
from mdl.custom_dataset import TFDataset
from eval.metric import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_entropy(y_, y, ns, nns, unigram):
    if ns == "uniform": return ns_uniform(y_, y, nns)
    if ns == "unigram": return ns_unigram(y_, y, unigram, nns)
    if ns == "unigram_b": return ns_unigram_mini_batch(y_, y, nns)
    return weighted(y_, y)

def weighted(logits, targets, pos_weight=2.5):
    return (-targets * torch.log(logits) * pos_weight + (1 - targets) * - torch.log(1 - logits)).sum()

def ns_uniform(logits, targets, neg_samples=5):
    targets = targets.squeeze(1)
    logits = logits.squeeze(1)
    random_samples = torch.zeros_like(targets)
    for b in range(targets.shape[0]):
        k_neg_idx = torch.randint(0, targets.shape[1], (neg_samples, ))
        cor_idx = torch.nonzero(targets[b].cpu(), as_tuple=True)[0]
        for idx in k_neg_idx:
            if idx not in cor_idx:
                random_samples[b][idx] = 1
    return (-targets * torch.log(logits) - random_samples * torch.log(1 - logits)).sum()

def ns_unigram(logits, targets, unigram, neg_samples=5):
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

def ns_unigram_mini_batch(logits, targets, neg_samples=5):
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

def learn(splits, indexes, vecs, params, output):

    layers = params['l']; learning_rate = params['lr']; batch_size = params['b']; num_epochs = params['e']; nns = params['nns']; ns = params['ns']
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
        X_train = vecs['skill'][splits['folds'][foldidx]['train'], :]; y_train = vecs['member'][splits['folds'][foldidx]['train']]
        X_valid = vecs['skill'][splits['folds'][foldidx]['valid'], :]; y_valid = vecs['member'][splits['folds'][foldidx]['valid']]

        # Building custom dataset
        training_matrix = TFDataset(X_train, y_train)
        validation_matrix = TFDataset(X_valid, y_valid)

        # Generating data loaders
        training_dataloader = DataLoader(training_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(validation_matrix, batch_size=batch_size, shuffle=True, num_workers=0)
        data_loaders = {"train": training_dataloader, "valid": validation_dataloader}

        # Initialize network
        model = FNN(input_size=input_size, output_size=output_size, param=params).to(device)
            
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
                    X = X.float().to(device=device)  # Get data to cuda if possible
                    y = y.float().to(device=device)
                    if phase == 'train':
                        model.train(True)  # scheduler.step()
                        # forward
                        optimizer.zero_grad()
                        y_ = model(X)
                        loss = cross_entropy(y_, y, ns, nns, unigram)
                        # backward
                        loss.backward()
                        # clip_grad_value_(model.parameters(), 1)
                        optimizer.step()
                        train_running_loss += loss.item()
                    else:  # valid
                        model.train(False)  # Set model to valid mode
                        y_ = model(X)
                        loss = cross_entropy(y_, y, ns, nns, unigram)
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

def test(nn_classname, model_path, splits, indexes, vecs, params, on_train_valid_set=False):
    if not os.path.isdir(model_path): raise Exception("The model does not exist!")
    # input_size = len(indexes['i2s'])
    input_size = vecs['skill'].shape[1]
    output_size = len(indexes['i2c'])

    X_test = vecs['skill'][splits['test'], :]
    y_test = vecs['member'][splits['test']]
    test_matrix = TFDataset(X_test, y_test)
    test_dl = DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

    for foldidx in splits['folds'].keys():
        model = nn_classname(input_size=input_size, output_size=output_size, param=params).to(device)
        model.load_state_dict(torch.load(f'{model_path}/state_dict_model_f{foldidx}.pt'))
        model.eval()

        for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
            if pred_set != 'test':
                X = vecs['skill'][splits['folds'][foldidx][pred_set], :]
                y = vecs['member'][splits['folds'][foldidx][pred_set]]
                matrix = TFDataset(X, y)
                dl = DataLoader(matrix, batch_size=params['b'], shuffle=True, num_workers=0)
            else:
                X = X_test; y = y_test; matrix = test_matrix; dl = test_dl

            torch.cuda.empty_cache()
            with torch.no_grad():
                y_pred = torch.empty(0, dl.dataset.output.shape[1])
                for x, y in dl:
                    x = x.to(device=device)
                    scores = model(x)
                    scores = scores.squeeze(1).cpu().numpy()
                    y_pred = np.vstack((y_pred, scores))

            torch.save(y_pred, f'{model_path}/f{foldidx}.{pred_set}.pred', pickle_protocol=4)

def evaluate(model_path, splits, vecs, on_train_valid_set=False, per_instance=False):
    if not os.path.isdir(model_path): raise Exception("The predictions do not exist!")
    y_test = vecs['member'][splits['test']]
    for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
        fold_mean = pd.DataFrame()
        plt.figure()
        for foldidx in splits['folds'].keys():
            if pred_set != 'test': Y = vecs['member'][splits['folds'][foldidx][pred_set]]
            else: Y = y_test
            Y_ = torch.load(f'{model_path}/f{foldidx}.{pred_set}.pred')
            df, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, per_instance)
            if per_instance: df.to_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.csv', float_format='%.15f')
            df_mean.to_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.mean.csv')
            with open(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'wb') as outfile:
                pickle.dump((fpr, tpr), outfile)
            fold_mean = pd.concat([fold_mean, df_mean], axis=1)
        #the last row is a list of roc values
        fold_mean.mean(axis=1).to_csv(f'{model_path}/{pred_set}.pred.eval.mean.csv')

def plot_roc(model_path, splits, on_train_valid_set=False):
    for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
        plt.figure()
        for foldidx in splits['folds'].keys():
            with open(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'rb') as infile: (fpr, tpr) = pickle.load(infile)
            # fpr, tpr = eval(pd.read_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.mean.csv', index_col=0).loc['roc'][0].replace('array', 'np.array'))
            plt.plot(fpr, tpr, label=f'micro-average fold{foldidx} on {pred_set} set', linestyle=':', linewidth=4)

        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC curves for {pred_set} set')
        plt.legend()
        plt.savefig(f'{model_path}/{pred_set}.roc.png', dpi=100, bbox_inches='tight')
        plt.show()

def main(splits, vecs, indexes, output, settings, cmd):
    output = f"{output}/t{vecs['skill'].shape[0]}.s{vecs['skill'].shape[1]}.m{vecs['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '')for k, v in settings.items() if v])}"
    if not os.path.isdir(output): os.makedirs(output)

    if 'train' in cmd: learn(splits, indexes, vecs, settings, output)
    if 'test' in cmd: test(FNN, output, splits, indexes, vecs, settings, on_train_valid_set=False)
    if 'eval' in cmd: evaluate(output, splits, vecs, on_train_valid_set=False, per_instance=False)
    if 'plot' in cmd: plot_roc(output, splits, on_train_valid_set=False)
