import logging, numpy as np
from tqdm import tqdm
log = logging.getLogger(__name__)

import pkgmgr as opentf
def calculate_metrics(Y, Y_, per_instance=False, metrics={'P_2,5,10', 'recall_2,5,10', 'ndcg_cut_2,5,10', 'map_cut_2,5,10'}):
    pd = opentf.install_import('pandas==2.0.0', 'pandas')
    pytrec_eval = opentf.install_import('pytrec-eval-terrier==0.5.7', 'pytrec_eval')
    # eval_met = dict(zip(metrics, [None]*len(metrics)))
    aucroc, fpr, tpr = calculate_auc_roc(Y, Y_)
    qrel = dict(); run = dict()
    log.info(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    with tqdm(total=Y.shape[0]) as pbar:
        for i, (y, y_) in enumerate(zip(Y, Y_)):
            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
            run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_)}
            pbar.update(1)
            # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
    log.info(f'Evaluating {metrics} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics).evaluate(run))
    log.info(f'Averaging ...\n')
    # df_mean = df.mean(axis=1).append(pd.Series([aucroc], index=['aucroc'])).to_frame('mean')
    df_mean = df.mean(axis=1).to_frame('mean')
    df_mean.loc['aucroc'] = aucroc
    return df if per_instance else None, df_mean, (fpr, tpr) # fpr, tpr is a long string that pandas truncate

def calculate_auc_roc(Y, Y_):
    scikit = opentf.install_import('scikit-learn==1.2.2', 'sklearn.metrics')
    log.info(f'Calculating roc_auc_score ...')
    auc = scikit.roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    log.info(f'Calculating roc_curve ...\n')
    fpr, tpr, _ = scikit.roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, fpr, tpr

# calculate skill_coverage for k = [2, 5, 10] for example
def calculate_skill_coverage(vecs, actual_skills, Y_, top_k=[2, 5, 10]):
    log.info(f"Calculating skill coverage for {Y_.shape[0]} predictions")
    if not isinstance(vecs['es_vecs'], np.ndarray): vecs['es_vecs'] = np.where(np.asarray(vecs['es_vecs'].todense()) > 0, 1, 0)
    skill_coverage = {}
    top_k_y_ = convert_to_one_hot(Y_, top_k) # convert the predicted experts to one-hot encodings based on top-k recommendations

    # we have to calculate skill_coverage for each value in the list top_k (2, 5 and 10 for example)
    for k in top_k:
        log.info(f"---- Calculating skc for k = {k}")
        Y_ = top_k_y_[k] # the 1-hot converted matrix for top k recommendations

        predicted_skills = np.where(np.dot(Y_, vecs['es_vecs']).astype(int) > 0, 1, 0)                                  # skill occurrence matrix of predicted members of shape (1 * |s|) for each row
        skills_overlap = ((predicted_skills & actual_skills) > 0).astype(int)                                           # overlap of skills in each row between predicted and actual
        skill_coverage[f'skc_{k}'] = np.average([r1.sum()/r2.sum() for r1,r2 in zip(skills_overlap,actual_skills)])     # avg coverage over all the predicted rows
        log.info(f"---- Calculated skc for k = {k}")

    return skill_coverage

# convert the top k expert prediction probabilities into 1-hot occurrences
# here top_k is a list of k's
def convert_to_one_hot(y_, top_k):
    print("Converting the prediction probabilities to 1-hot predictions")
    top_k_matrices = {}

    for k in top_k:
        print(f"-------- Converting for k = {k}")
        result = np.zeros_like(y_)

        for i in tqdm(range(y_.shape[0])):
            top_k_indices = np.argsort(y_[i])[-k:] # get the indices of the top k values
            result[i, top_k_indices] = 1 # set the top k values to 1

        top_k_matrices[k] = result
        print(f"-------- Converted for k = {k}\n")

    return top_k_matrices # |test_instances| * |num_test_instance_experts| for each k in top_k