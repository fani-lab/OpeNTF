import os, pickle
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn

from eval.metric import *

class Ntf(nn.Module):
    def __init__(self):
        super(Ntf, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def learn(self, splits, indexes, vecs, params, output): pass
    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False): pass

    def evaluate(self, model_path, splits, vecs, on_train_valid_set=False, per_instance=False):
        if not os.path.isdir(model_path): raise Exception("The predictions do not exist!")
        y_test = vecs['member'][splits['test']]
        for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
            fold_mean = pd.DataFrame()
            for foldidx in splits['folds'].keys():
                if pred_set != 'test':
                    Y = vecs['member'][splits['folds'][foldidx][pred_set]]
                else:
                    Y = y_test
                Y_ = torch.load(f'{model_path}/f{foldidx}.{pred_set}.pred')
                df, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, per_instance)
                if per_instance: df.to_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.csv', float_format='%.15f')
                df_mean.to_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.mean.csv')
                with open(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'wb') as outfile:
                    pickle.dump((fpr, tpr), outfile)
                fold_mean = pd.concat([fold_mean, df_mean], axis=1)
            # the last row is a list of roc values
            fold_mean.mean(axis=1).to_frame('mean').to_csv(f'{model_path}/{pred_set}.pred.eval.mean.csv')

    def plot_roc(self, model_path, splits, on_train_valid_set=False):
        for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
            plt.figure()
            for foldidx in splits['folds'].keys():
                with open(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'rb') as infile: (
                    fpr, tpr) = pickle.load(infile)
                # fpr, tpr = eval(pd.read_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.mean.csv', index_col=0).loc['roc'][0].replace('array', 'np.array'))
                plt.plot(fpr, tpr, label=f'micro-average fold{foldidx} on {pred_set} set', linestyle=':', linewidth=4)

            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f'ROC curves for {pred_set} set')
            plt.legend()
            plt.savefig(f'{model_path}/{pred_set}.roc.png', dpi=100, bbox_inches='tight')
            plt.show()

    def run(self, splits, vecs, indexes, output, settings, cmd):
        output = f"{output}/t{vecs['skill'].shape[0]}.s{vecs['skill'].shape[1]}.m{vecs['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings.items() if v])}"
        if not os.path.isdir(output): os.makedirs(output)

        if 'train' in cmd: self.learn(splits, indexes, vecs, settings, output)
        if 'test' in cmd: self.test(output, splits, indexes, vecs, settings, on_train_valid_set=False)
        if 'eval' in cmd: self.evaluate(output, splits, vecs, on_train_valid_set=False, per_instance=False)
        if 'plot' in cmd: self.plot_roc(output, splits, on_train_valid_set=False)



