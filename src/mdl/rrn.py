import pickle
import subprocess, os
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

from eval.metric import *
from mdl.ntf import Ntf

class Rrn(Ntf):
    def __init__(self):
        super(Ntf, self).__init__()

    def prepare_data(self, teamsvec, ind, model_path, with_zeros=True):
        if with_zeros:
            self.prepare_data_with_zeros(teamsvec, ind, model_path)   
        else:
            self.prepare_data_without_zeros(teamsvec, ind, model_path)  
    
    def prepare_data_without_zeros(self, teamsvec, ind, model_path):
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
        with open(f"{model_path}/train.data", "w") as file1:
            file1.write(str(skill.shape[1])+"\n")
            file1.write(str(member.shape[1])+"\n")    
            for i in range(1, len(ind['i2y'])-2):
                colab = skill[ind['i2y'][i-1][0]:ind['i2y'][i][0]].T @ member[ind['i2y'][i-1][0]:ind['i2y'][i][0]]
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "train", "time_stamp": {i}, "rating": 1}}\n'
                    file1.write(instance)
            
            with open(f"{model_path}/test.data", "w") as file2:
                colab = skill[ind['i2y'][-2][0]:].T @ member[ind['i2y'][-2][0]:]
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-2}, "rating": 1}}\n'
                    file2.write(instance)
    
    def prepare_data_with_zeros(self, teamsvec, ind, model_path):
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
        with open(f"{model_path}/train_with_zero.data", "w") as file1:
            file1.write(str(skill.shape[1])+"\n")
            file1.write(str(member.shape[1])+"\n")    
            for i in range(1, len(ind['i2y'])-2):
                n_teams_per_year = ind['i2y'][i][0] - ind['i2y'][i-1][0]
                compl_skill = np.ones((n_teams_per_year, skill.shape[1]), int)
                compl_member = np.ones((n_teams_per_year, member.shape[1]), int)
                compl_skill[skill[ind['i2y'][i-1][0]:ind['i2y'][i][0]].nonzero()] = 0
                compl_member[member[ind['i2y'][i-1][0]:ind['i2y'][i][0]].nonzero()] = 0

                colab = skill[ind['i2y'][i-1][0]:ind['i2y'][i][0]].T @ member[ind['i2y'][i-1][0]:ind['i2y'][i][0]]
                no_colab = lil_matrix(compl_skill.T @ compl_member)

                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "train", "time_stamp": {i}, "rating": 1}}\n'
                    file1.write(instance)
                no_rows, no_cols = no_colab.nonzero()
                for row, col in zip(no_rows, no_cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "train", "time_stamp": {i}, "rating": 0}}\n'
                    file1.write(instance)
            
            with open(f"{model_path}/test_with_zero.data", "w") as file2:
                n_teams_test_set = skill.shape[0] - ind['i2y'][-2][0]
                compl_skill = np.ones((n_teams_test_set, skill.shape[1]), int)
                compl_member = np.ones((n_teams_test_set, member.shape[1]), int)
                
                compl_skill[skill[ind['i2y'][-2][0]:].nonzero()] = 0
                compl_member[member[ind['i2y'][-2][0]:].nonzero()] = 0
                colab = skill[ind['i2y'][-2][0]:].T @ member[ind['i2y'][-2][0]:]
                no_colab = lil_matrix(compl_skill.T @ compl_member)
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-2}, "rating": 1}}\n'
                    file2.write(instance)
                no_rows, no_cols = no_colab.nonzero()
                for row, col in zip(no_rows, no_cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-2}, "rating": 0}}\n'
                    file2.write(instance)

    def learn(self, model_path, with_zeros=True):
        if with_zeros:
            wz = '_with_zero'
        else:
            wz = ''
        cli_cmd = 'python ../baseline/rrn/src/rrn_main.py '
        cli_cmd += f'--train_file {model_path}/train{wz}.data '
        cli_cmd += f'--test_val_file {model_path}/test{wz}.data '
        cli_cmd += '--mf_model_file ../baseline/rrn/model/imdb_15core_M.model40.npy'
        print(f'{cli_cmd}')
        subprocess.Popen(shlex.split(cli_cmd)).wait()
    
    def test(self):
        pass # test already done in learn

    def eval(self, splits, teamsvec, path):
        results = pd.read_csv(f'{path}/pred', sep='\t', header=None)
        results.rename(columns={0:'skill', 1:'member', 2:'gt', 3:'pred'}, inplace=True)
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
                
        Y = member[splits['test']]
        Y_ = np.zeros((len(splits['test']), member.shape[1]))
        for i in range(len(splits['test'])):
            x_test = skill[splits['test']][i].nonzero()[1]
            pred = results.loc[results['skill'].isin(x_test)]
            pred = pred.groupby(['member'], as_index=False).mean()
            for ind, row in pred.iterrows():
                Y_[i][int(row['member'])] = row['pred']

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

        with_zero = True

        if 'train' in cmd:
            self.prepare_data(vecs, indexes, model_path, with_zeros=with_zero)
            self.learn(model_path, with_zeros=with_zero)
        if 'test' in cmd: self.test()
        if 'eval' in cmd: self.eval(splits, vecs, model_path)
        if 'plot' in cmd: self.plot_roc(model_path)