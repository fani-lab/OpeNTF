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
    for i, (y, y_) in enumerate(zip(Y, Y_)):
        qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
        run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j,v in enumerate(y_)}
        # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
    print(f'Evaluating {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    print(f'Averaging ...')
    df_mean = df.mean(axis=1).append(pd.Series([aucroc], index=['aucroc'])).to_frame('mean')
    return df if per_instance else None, df_mean, (fpr, tpr) # fpr, tpr is a long string that pandas truncate

def calculate_auc_roc(Y, Y_):
    auc = roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    fpr, tpr, _ = roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, fpr, tpr

