import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, ndcg_score
from sklearn.metrics import roc_curve
import pytrec_eval
import json
import pandas as pd

def calculate_metrics(Y, Y_, metrics=None):
    # eval_met = dict(zip(metrics, [None]*len(metrics)))
    aucroc = roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    fpr, tpr, _ = roc_curve(Y.toarray().ravel(), Y_.ravel())

    qrel = dict(); run = dict()
    for i, (y, y_) in enumerate(zip(Y, Y_)):
        qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
        run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j,v in enumerate(y_)}
        # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}

    p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_2,4,6,8,10', 'recall_2,4,6,8,10', 'ndcg_cut_2,4,6,8,10', 'map_cut_2,4,6,8,10'})
    df = pd.DataFrame.from_dict(p_evaluator.evaluate(run))
    return df, df.mean(axis=1).append(pd.Series([aucroc, (fpr, tpr)], index=['aucroc', 'roc']))

def plot_ndcg_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        ndcg_at_k = []
        for k in range(10, 60, 10):
        # for k in range(2, 12, 2):
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)

                scores = model(x)

                y = y.squeeze(1).cpu().numpy()
                scores = scores.squeeze(1).cpu().numpy()

                y_true = np.vstack((y_true, y))
                y_pred = np.vstack((y_pred, scores))



            ndcg_at_k.append(ndcg_score(y_true, y_pred, k=k))
        return ndcg_at_k

def plot_recall_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        r_at_k = []
        for k in range(10, 60, 10):
        # for k in range(2, 12, 2):
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device).squeeze(1).cpu().numpy()
                topk_scores = torch.zeros(y.shape[0], y.shape[1])

                scores = model(x)
                topk_idx = torch.topk(scores, k)[1].squeeze(1).cpu().numpy()
                for i in range(topk_idx.shape[0]):
                    for j in topk_idx[i]:
                        topk_scores[i][j] = 1
                y_true = np.vstack((y_true, y))
                y_pred = np.vstack((y_pred, topk_scores))


            r_at_k.append(recall_score(y_true.ravel(), y_pred.ravel()))
        return r_at_k

def plot_precision_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        p_at_k = []
        for k in range(10, 60, 10):
        # for k in range(2, 12, 2):
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device).squeeze(1).cpu().numpy()
                topk_scores = torch.zeros(y.shape[0], y.shape[1])

                scores = model(x)
                topk_idx = torch.topk(scores, k)[1].squeeze(1).cpu().numpy()
                for i in range(topk_idx.shape[0]):
                    for j in topk_idx[i]:
                        topk_scores[i][j] = 1

                y_true = np.vstack((y_true, y))
                y_pred = np.vstack((y_pred, topk_scores))

            p_at_k.append(precision_score(y_true.ravel(), y_pred.ravel()))
        return p_at_k

def plot_roc(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # scores = torch.sigmoid(scores)
            # scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        return fpr, tpr

def roc_auc(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # scores = torch.sigmoid(scores)
            # scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        auc = roc_auc_score(y_true, y_pred, average='micro', multi_class="ovr")
        return str(auc)

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
