import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve
import pytrec_eval
import json
import pandas as pd

# def calculate_metrics(Y, Y_, metrics=None):
#     # eval_met = dict(zip(metrics, [None]*len(metrics)))
#     aucroc = roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
#     fpr, tpr, _ = roc_curve(Y.toarray().ravel(), Y_.ravel())
#
#     qrel = dict(); run = dict()
#     for i, (y, y_) in enumerate(zip(Y, Y_)):
#         qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
#         run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j,v in enumerate(y_)}
#         # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
#
#     p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_2,4,6,8,10', 'recall_2,4,6,8,10', 'ndcg_cut_2,4,6,8,10', 'map_cut_2,4,6,8,10'})
#     df = pd.DataFrame.from_dict(p_evaluator.evaluate(run))
#     return df, df.mean(axis=1).append(pd.Series([aucroc, (fpr, tpr)], index=['aucroc', 'roc']))

def calculate_auc_roc(Y, Y_):
    auc = roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    fpr, tpr, _ = roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, fpr, tpr

# def calculate_ranking_metrics(Y, Y_, metric='P'):
#     qrel = dict();
#     run = dict()
#     for i, (y, y_) in enumerate(zip(Y, Y_)):
#         qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
#         run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_)}
#         # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
#
#     p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {f'{metric}_2,4,6,8,10'})
#     df = pd.DataFrame.from_dict(p_evaluator.evaluate(run))
#     return df, df.mean(axis=1)

def calculate_ranking_metrics(Y, Y_, metrics=['P', 'recall', 'ndcg_cut', 'map_cut']):
    qrel = dict();
    run = dict()
    for i, (y, y_) in enumerate(zip(Y, Y_)):
        qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
        run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_)}
        # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
    for metric in metrics:
        met = f'{metric}_2,4,6,8,10'
        p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {met})
        df = pd.DataFrame.from_dict(p_evaluator.evaluate(run))
        yield metric, df, df.mean(axis=1)

def cls_rep(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        f1 = classification_report(y_true, y_pred, zero_division=0)
        return f1
