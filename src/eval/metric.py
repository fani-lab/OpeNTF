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

# calculate skill_coverage for k = [2, 5, 10] for example
def calculate_skill_coverage(vecs, Y, Y_, top_k):

    if not isinstance(vecs['es_vecs'], np.ndarray):
        vecs['es_vecs'] = np.where(np.asarray(vecs['es_vecs'].todense()) > 0, 1, 0)
    skill_coverage = {}
    top_k_y_ = convert_to_one_hot(Y_, top_k) # convert the predicted experts to one-hot encodings based on top-k recommendations

    # we have to calculate skill_coverage for each value in the list top_k (2, 5 and 10 for example)
    for k in top_k:
        Y_ = top_k_y_[k] # the 1-hot converted matrix for top k recommendations

        predicted_skills = np.where(np.dot(Y_, vecs['es_vecs']).astype(int) > 0, 1, 0) # skill occurrence matrix of predicted members of shape (1 * |s|) for each row
        actual_skills = np.where(np.dot(np.asarray(Y.todense()), vecs['es_vecs']).astype(int) > 1, 1, 0) # skills of actual members of shape (1 * |s|) for each row

        n_skills = [(row > 0).astype(int).sum() for row in actual_skills] # total number of skills of the actual members

        skills_overlap = ((predicted_skills & actual_skills) > 0).astype(int) # overlap of skills in each row between predicted and actual

        n_skills_covered = [row.sum() for row in skills_overlap] # total number of skills common between actual and predicted members for each prediction row
        skill_coverage[f'skc_{k}'] = np.average([(n_skills_covered[i] / n_skills[i]) for i in range(len(n_skills))]) # avg coverage over all the predicted rows

    return skill_coverage

# convert the top k expert prediction probabilities into 1-hot occurrences
# here top_k is a list of k's
def convert_to_one_hot(y_, top_k):
    top_k_matrices = {}

    for k in top_k:
        result = np.zeros_like(y_)

        for i in range(y_.shape[0]):
            top_k_indices = np.argsort(y_[i])[-k:] # get the indices of the top k values
            result[i, top_k_indices] = 1 # set the top k values to 1

        top_k_matrices[k] = result

    return top_k_matrices # |test_instances| * |num_test_instance_experts| for each k in top_k