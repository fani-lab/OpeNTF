import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve
import pytrec_eval
import json
import pandas as pd

def calculate_metrics(Y, Y_, per_instance=False, metrics={'P_2,5,10', 'recall_2,5,10', 'ndcg_cut_2,5,10', 'map_cut_2,5,10'}):
    # eval_met = dict(zip(metrics, [None]*len(metrics)))
    aucroc, fpr, tpr = calculate_auc_roc(Y, Y_)

    qrel = dict(); run = dict()
    print(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    from tqdm import tqdm
    with tqdm(total=Y.shape[0]) as pbar:
        for i, (y, y_) in enumerate(zip(Y, Y_)):
            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
            run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_)}
            pbar.update(1)
            # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
    print(f'Evaluating {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    print(f'Averaging ...\n')
    # df_mean = df.mean(axis=1).append(pd.Series([aucroc], index=['aucroc'])).to_frame('mean')
    df_mean = df.mean(axis=1).to_frame('mean')
    df_mean.loc['aucroc'] = aucroc
    return df if per_instance else None, df_mean, (fpr, tpr) # fpr, tpr is a long string that pandas truncate

def calculate_auc_roc(Y, Y_):
    print(f'\nCalculating roc_auc_score ...')
    auc = roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    print(f'Calculating roc_curve ...\n')
    fpr, tpr, _ = roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, fpr, tpr

def calculate_skill_coverage(vecs, Y, Y_):

    predicted_skills = np.dot(Y_, vecs['es_vecs']) # skill occurrence matrix of predicted members of shape (1 * |s|)
    actual_skills = np.dot(Y, vecs['es_vecs']) # skills of actual members of shape (1 * |s|)

    n_skills = (actual_skills >= 1).astype(int).sum() # total number of skills of the actual members
    n_skills_covered = ((predicted_skills & actual_skills) >= 1).astype(int).sum() # total number of skills common between actual and predicted members
    skill_coverage = (n_skills_covered / n_skills) * 100

    return skill_coverage