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
datasets += ['../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3']
# datasets += ['../data/preprocessed/uspt/patent.tsv.filtered.mt75.ts3']


def reformulate_input_to_caser(datasets):
    for dataset in datasets:
        dname = dataset.split('/')[-2]
        with open(f'{dataset}/indexes.pkl', 'rb') as infile:
            ind = pickle.load(infile)
        with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile:
            teamsvec = pickle.load(infile)
        
        skill = lil_matrix(teamsvec['skill'])
        member = lil_matrix(teamsvec['member'])
        with open(f"{dataset}/train.txt", "w") as file1:
            for i in range(1, len(ind['i2y'])-2):
                colab = skill[ind['i2y'][i-1][0]:ind['i2y'][i][0]].T @ member[ind['i2y'][i-1][0]:ind['i2y'][i][0]]
                rows, cols = colab.nonzero()
                for row, col in zip(rows, cols):
                    instance = f'{row} {col} 1\n'
                    file1.write(instance)
        with open(f"{dataset}/test.txt", "w") as file2:
            colab = skill[ind['i2y'][-2][0]:].T @ member[ind['i2y'][-2][0]:]
            rows, cols = colab.nonzero()
            for row, col in zip(rows, cols):
                instance = f'{row} {col} 1\n'
                file2.write(instance)
        
    
# reformulate_input_to_caser(datasets)

def transform_caser_output(datasets):
    for dataset in datasets:
        dname = dataset.split('/')[-2]
        results = pd.read_csv(f'{dataset}/caser_{dname}_lr0.0001_dp0.001_TL22.csv')
        # results.rename(columns={0:'skill', 1:'member', 2:'gt', 3:'pred'}, inplace=True)
        with open(f'{dataset}/indexes.pkl', 'rb') as infile:
            ind = pickle.load(infile)
        with open(f'{dataset}/teamsvecs.pkl', 'rb') as infile2:   
            teamsvec = pickle.load(infile2)
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
                # We need to use int(row['member'])-1 because the caser code creates a new class for 0 padding for convolutions and hence the index of predicted classes in their result need to be be diminished by 1
                Y_[i][int(row['member'])-1] = row['pred'] 
        _, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, False)
        df_mean.to_csv(f'{dataset}/caser_{dname}_lr0.0001_dp0.001_TL22.pred.eval.mean.csv')
        with open(f'{dataset}/caser_{dname}_lr0.0001_dp0.001_TL22.pred.eval.roc.pkl', 'wb') as outfile:
            pickle.dump((fpr, tpr), outfile)
        plt.figure()
        plt.plot(fpr, tpr, label=f'micro-average on test set', linestyle=':', linewidth=4)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC curves for test set')
        plt.legend()
        plt.savefig(f'{dataset}/caser_{dname}_lr0.0001_dp0.001_TL22.roc.png', dpi=100, bbox_inches='tight')
        plt.show()

transform_caser_output(datasets)