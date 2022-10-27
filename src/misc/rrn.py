import sys
sys.path.extend(["./"])
import pickle
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np 
import pandas as pd
from eval.metric import *
import json

datasets = []
# datasets += ['../data/preprocessed/dblp/toy.dblp.v12.json']
# datasets += ['../data/preprocessed/imdb/toy.title.basics.tsv']
# datasets += ['../data/preprocessed/uspt/toy.patent.tsv']
# datasets += ['../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3']


def reformulate_input_to_rrn(datasets):
    for dataset in datasets:
        dname = dataset.split('/')[-2]
        with open(f'{dataset}/indexes.pkl', 'rb') as infile:
            ind = pickle.load(infile)
        with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile2:
            teamsvec = pickle.load(infile2)
        
        id = lil_matrix(teamsvec['id'])
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
        with open(f"{dname}_train.data", "w") as file1:
            file1.write(str(skill.shape[1])+"\n")
            file1.write(str(member.shape[1])+"\n")    
            for i in range(1, len(ind['i2y'])-1):
                colab = skill[ind['i2y'][i-1][0]:ind['i2y'][i][0]].T @ member[ind['i2y'][i-1][0]:ind['i2y'][i][0]]
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "train", "time_stamp": {i}, "rating": 1}}\n'
                    file1.write(instance)
            
            with open(f"{dname}_test.data", "w") as file2:
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-1}, "rating": 1}}\n'
                    file2.write(instance)

def reformulate_input_to_rrn_with_zero(datasets):
    for dataset in datasets:
        dname = dataset.split('/')[-2]
        with open(f'{dataset}/indexes.pkl', 'rb') as infile:
            ind = pickle.load(infile)
        with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile2:
            teamsvec = pickle.load(infile2)
        
        id = lil_matrix(teamsvec['id'])
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
        with open(f"{dname}_train_with_zero.data", "w") as file1:
            file1.write(str(skill.shape[1])+"\n")
            file1.write(str(member.shape[1])+"\n")    
            for i in range(1, len(ind['i2y'])-1):
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
            
            with open(f"{dname}_test_with_zero.data", "w") as file2:
                n_teams_test_set = skill.shape[0] - ind['i2y'][-2][0]
                compl_skill = np.ones((n_teams_test_set, skill.shape[1]), int)
                compl_member = np.ones((n_teams_test_set, member.shape[1]), int)
                
                compl_skill[skill[ind['i2y'][-2][0]:].nonzero()] = 0
                compl_member[member[ind['i2y'][-2][0]:].nonzero()] = 0
                colab = skill[ind['i2y'][-2][0]:].T @ member[ind['i2y'][-2][0]:]
                no_colab = lil_matrix(compl_skill.T @ compl_member)
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-1}, "rating": 1}}\n'
                    file2.write(instance)
                no_rows, no_cols = no_colab.nonzero()
                for row, col in zip(no_rows, no_cols):
                    instance = f'{{"u_idx": {row}, "m_idx": {col}, "split": "test", "time_stamp": {len(ind["i2y"])-1}, "rating": 0}}\n'
                    file2.write(instance)

# reformulate_input_to_rrn(datasets)


def transform_rrn_output(datasets):
    for dataset in datasets:
        dname = dataset.split('/')[-2]
        results = pd.read_csv(f'{dataset}/{dname}_pred', sep='\t', header=None)
        results.rename(columns={0:'skill', 1:'member', 2:'gt', 3:'pred'}, inplace=True)
        with open(f'{dataset}/indexes.pkl', 'rb') as infile:
            ind = pickle.load(infile)
        with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile2:   
            teamsvec = pickle.load(infile2)
        id = lil_matrix(teamsvec['id'])
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
                
        with open(f'{dataset}/splits.json') as infile3:
            splits = json.load(infile3)
        Y = member[splits['test']]
        Y_ = np.zeros((len(splits['test']), member.shape[1]))
        for i in range(len(splits['test'])):
            x_test = skill[splits['test']][i].nonzero()[1]
            pred = results.loc[results['skill'].isin(x_test)]
            pred = pred.groupby(['member'], as_index=False).mean()
            for ind, row in pred.iterrows():
                Y_[i][int(row['member'])] = row['pred']

        _, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, False)
        df_mean.to_csv(f'{dataset}/{dname}.pred.eval.mean.csv')
        with open(f'{dataset}/{dname}.pred.eval.roc.pkl', 'wb') as outfile:
            pickle.dump((fpr, tpr), outfile)
        plt.figure()
        plt.plot(fpr, tpr, label=f'micro-average on test set', linestyle=':', linewidth=4)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC curves for test set')
        plt.legend()
        plt.savefig(f'{dataset}/{dname}.roc.png', dpi=100, bbox_inches='tight')
        plt.show()

# transform_rrn_output(datasets)