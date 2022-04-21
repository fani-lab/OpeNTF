import os, time, pickle, json, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import leaky_relu
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from eval.metric import *
from mdl.bnn import Bnn
from mdl.custom_dataset import TFDataset
from mdl.temporal_dataset import TTFDataset

from cmn.team import Team

class TBnn(Bnn):
    def __init__(self):
        super(TBnn, self).__init__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def learn(self, splits, indexes, vecs, params, output):
        layers = params['l']
        learning_rate = params['lr']
        batch_size = params['b']
        num_epochs = params['e']
        nns = params['nns']
        ns = params['ns']
        s = params['s']
        # input_size = len(indexes['i2s'])
        input_size = vecs['skill'].shape[1]
        output_size = len(indexes['i2c'])

        unigram = Team.get_unigram(vecs['member'])

        # Prime a dict for train and valid loss
        training_loss = {'train': []}

        start_time = time.time()
        
        # Retrieving the folds
        X_train = vecs['skill'][splits['train'], :]
        y_train = vecs['member'][splits['train']]
        list_n_teams_per_year = vecs['year_idx']

        list_n_teams_per_year = [x for x in list_n_teams_per_year if x < splits['train'][-1]]
        list_n_teams_per_year.append(splits['train'][-1])

        # Building custom dataset
        training_matrix = TTFDataset(X_train, y_train, list_n_teams_per_year)

        # Generating data loaders
        training_dataloader = DataLoader(training_matrix, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize network
        self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_loss_values = []
        # Train Network
        for batch_idx, (X, y) in enumerate(training_dataloader):
            train_running_loss = 0.0
            for epoch in range(num_epochs):
                epoch_time = time.time()
            
                torch.cuda.empty_cache()
                X = X.float().to(device=self.device)  # Get data to cuda if possible
                y = y.float().to(device=self.device)

                self.train(True)  # scheduler.step()
                # forward
                optimizer.zero_grad()
                y_ = self.forward(X)
                layer_loss, y_ = self.sample_elbo(X.squeeze(0), y.squeeze(0), s)
                loss = self.cross_entropy(y_.squeeze(0).to(self.device), y.squeeze(0), ns, nns, unigram) + layer_loss / batch_size
                # backward
                loss.backward()
                # clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                train_running_loss += loss.item()
                print(
                    f'Minibatch {batch_idx}/{len(list_n_teams_per_year) - 2}, Epoch {epoch}/{num_epochs - 1}, Phase train'
                    f', Running Loss of training {loss.item()}'
                    f", Epoch time {time.time() - epoch_time}, Overall {time.time() - start_time} "
                )
            
            train_loss_values.append(train_running_loss / num_epochs)
            torch.save(self.state_dict(), f"{output}/state_dict_model.b{batch_idx}.pt", pickle_protocol=4)

        model_path = f"{output}/state_dict_model.pt"

        # Save
        torch.save(self.state_dict(), model_path, pickle_protocol=4)

        training_loss['train'] = train_loss_values

        print(f"It took {time.time() - start_time} to train the model.")
        with open(f"{output}/train_valid_loss.json", 'w') as outfile:
            json.dump(training_loss, outfile)
            plt.figure()
            plt.plot(training_loss['train'], label='Training Loss')
            plt.legend(loc='upper right')
            plt.title('Training loss')
            plt.savefig(f'{output}/training_loss.png', dpi=100, bbox_inches='tight')
            plt.show()

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False):
        if not os.path.isdir(model_path): raise Exception("The model does not exist!")
        input_size = vecs['skill'].shape[1]
        output_size = len(indexes['i2c'])

        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = TFDataset(X_test, y_test)
        test_dl = DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

        modelfiles = [f'{model_path}/state_dict_model.pt']
        if per_epoch: modelfiles = [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match(f'state_dict_model.b\d+.pt', _)]

        for modelfile in modelfiles:
            self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)
            self.load_state_dict(torch.load(modelfile))
            self.eval()

            for pred_set in (['test', 'train'] if on_train_valid_set else ['test']):
                if pred_set != 'test':
                    X = vecs['skill'][splits[pred_set], :]
                    y = vecs['member'][splits[pred_set]]
                    matrix = TFDataset(X, y)
                    dl = DataLoader(matrix, batch_size=params['b'], shuffle=False, num_workers=0)
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
                batch = modelfile.split('.')[-2] + '.' if per_epoch else ''
                torch.save(y_pred, f'{model_path}/{pred_set}.{batch}pred', pickle_protocol=4)

    def evaluate(self, model_path, splits, vecs, on_train_valid_set=False, per_instance=False, per_epoch=False):
        if not os.path.isdir(model_path): raise Exception("The predictions do not exist!")
        y_test = vecs['member'][splits['test']]
        for pred_set in (['test', 'train'] if on_train_valid_set else ['test']):
            batch_re = 'state_dict_model.e\d+' if per_epoch else 'state_dict_model.pt'
            predfiles = [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match(batch_re, _)]

            for e in range(len(predfiles)):
                batch = f'b{e}.' if per_epoch else ""
                if pred_set != 'test':
                    Y = vecs['member'][splits[pred_set]]
                else:
                    Y = y_test
                Y_ = torch.load(f'{model_path}/{pred_set}.{batch}pred')
                df, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, per_instance)
                if per_instance: df.to_csv(f'{model_path}/{pred_set}.pred.eval.csv', float_format='%.15f')
                df_mean.to_csv(f'{model_path}/{pred_set}.{batch}pred.eval.mean.csv')
                with open(f'{model_path}/{pred_set}.{batch}pred.eval.roc.pkl', 'wb') as outfile:
                    pickle.dump((fpr, tpr), outfile)

    def plot_roc(self, model_path, splits, on_train_valid_set=False, per_epoch=False):
        for pred_set in (['test', 'train'] if on_train_valid_set else ['test']):
            batch_re = 'state_dict_model.e\d+' if per_epoch else 'state_dict_model.pt'
            predfiles = [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match(batch_re, _)]

            for e in range(len(predfiles)):
                batch = f'b{e}.' if per_epoch else ""
                plt.figure()
                with open(f'{model_path}/{pred_set}.{batch}pred.eval.roc.pkl', 'rb') as infile: (fpr, tpr) = pickle.load(infile)
                plt.plot(fpr, tpr, label=f'micro-average on {pred_set} set {batch}', linestyle=':', linewidth=4)
                plt.xlabel('false positive rate')
                plt.ylabel('true positive rate')
                plt.title(f'ROC curves for {pred_set} set {batch}')
                plt.legend()
                plt.savefig(f'{model_path}/{pred_set}.{batch}roc.png', dpi=100, bbox_inches='tight')
                plt.show()