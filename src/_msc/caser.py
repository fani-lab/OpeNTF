import pickle
import subprocess, os
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

from eval.metric import *
from mdl.ntf import Ntf

class Caser(Ntf):
    def __init__(self, step_ahead=2):
        super(Ntf, self).__init__()
        self.step_ahead = step_ahead

    def prepare_data(self, vecs, indexes, model_path):
        skill = lil_matrix(vecs['skill'])
        member = lil_matrix(vecs['member'])
        with open(f"{model_path}/train.txt", "w") as file1:
            for i in range(1, len(indexes['i2y'])-(self.step_ahead)):
                colab = skill[indexes['i2y'][i-1][0]:indexes['i2y'][i][0]].T @ member[indexes['i2y'][i-1][0]:indexes['i2y'][i][0]]
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{row} {col} 1\n'
                    file1.write(instance)
        with open(f"{model_path}/test.txt", "w") as file2:
            colab = skill[indexes['i2y'][-(self.step_ahead)][0]:].T @ member[indexes['i2y'][-(self.step_ahead)][0]:]
            rows, cols = colab.nonzero()
            for row, col in zip(rows, cols):
                instance = f'{row} {col} 1\n'
                file2.write(instance)

    def learn(self, path):
        cli_cmd = 'python ../baseline/caser_pytorch/train_caser.py '
        cli_cmd += f'--train_root {path}/train.txt '
        cli_cmd += f'--test_root {path}/test.txt'
        print(f'{cli_cmd}')
        subprocess.Popen(shlex.split(cli_cmd)).wait()
    
    def test(self):
        pass # test already done in learn

    def eval(self, splits, vecs, path):
        results = pd.read_csv(f'{path}/pred.csv')
        skill = lil_matrix(vecs['skill'])
        member = lil_matrix(vecs['member'])
    
        Y = member[splits['test']]
        Y_ = np.zeros((len(splits['test']), member.shape[1]))
        for i in range(len(splits['test'])):
            x_test = skill[splits['test']][i].nonzero()[1]
            pred = results.loc[results['skill'].isin(x_test)]
            pred = pred.groupby(['member'], as_index=False).mean()
            for ind, row in pred.iterrows():
                # We need to use int(row['member'])-1 because the caser code creates a new class for 0 padding for convolutions and hence the index of predicted classes in their result need to be be diminished by 1
                Y_[i][int(row['member'])-1] = row['pred']
        Y_ = np.nan_to_num(Y_)
        _, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, False)
        df_mean.to_csv(f'{path}/pred.eval.mean.csv')
        with open(f'{path}/pred.eval.roc.pkl', 'wb') as outfile:
            pickle.dump((fpr, tpr), outfile)
    
    def plot_roc(self, model_path):
        with open(f'{model_path}/pred.eval.roc.pkl', 'rb') as infile: (fpr, tpr) = pickle.load(infile)
        plt.figure()
        plt.plot(fpr, tpr, label=f'micro-average on test set', linestyle=':', linewidth=4)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC curves for test set')
        plt.legend()
        plt.savefig(f'{model_path}/roc.png', dpi=100, bbox_inches='tight')
        plt.show()

    def run(self, splits, vecs, indexes, output, settings, cmd):
        team_count = vecs['skill'].shape[0]
        skill_count = vecs['skill'].shape[1]
        member_count = vecs['member'].shape[1]

        model_path = f"{output}/t{team_count}.s{skill_count}.m{member_count}"
        if not os.path.isdir(output): os.makedirs(output)
        if not os.path.isdir(model_path): os.makedirs(model_path)
        with open(f'{model_path}/indexes.pkl', "wb") as outfile: pickle.dump(indexes, outfile) 

        if 'train' in cmd:
            self.prepare_data(vecs, indexes, model_path)
            self.learn(model_path)
        if 'test' in cmd: self.test()
        if 'eval' in cmd: self.eval(splits, vecs, model_path)
        if 'plot' in cmd: self.plot_roc(model_path)