import logging, numpy as np
from tqdm import tqdm
log = logging.getLogger(__name__)

import pkgmgr as opentf
def calculate_metrics(Y, Y_, per_instance=False, metrics={'trec': ['P_2,5', 'recall_2,5', 'ndcg_cut_2,5'], 'other': ['aucroc']}):
    pd = opentf.install_import('pandas==2.0.0', 'pandas')
    pytrec_eval = opentf.install_import('pytrec-eval-terrier==0.5.7', 'pytrec_eval')
    qrel = dict(); run = dict()
    log.info(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    with tqdm(total=Y.shape[0]) as pbar:
        for i, (y, y_) in enumerate(zip(Y, Y_)):
            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in y.nonzero()[1]}
            run['q' + str(i)] = {'d' + str(j): int(np.round(v * 100)) for j, v in enumerate(y_)}
            pbar.update(1)
            # run['q' + str(i)] = {'d' + str(idx): int(np.round(y_[idx] * 10)) for idx in np.where(y_ > 0.5)[0]}
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, set(metrics.trec)).evaluate(run))
    df_mean = df.mean(axis=1).to_frame('mean')
    return df if per_instance else None, df_mean

def calculate_auc_roc(Y, Y_):
    scikit = opentf.install_import('scikit-learn==1.2.2', 'sklearn.metrics')
    auc = scikit.roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    fpr, tpr, _ = scikit.roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, (fpr, tpr)

def calculate_skill_coverage(X, Y_, expertskillvecs, per_instance=False, topks='2,5,10'):#skillcoveragevecs: ExS, X: BatchxS, Y_: BatchxE
    B, E = Y_.shape #batches of output expert recommendations for each team
    assert not np.any(np.all(X.toarray() != 0, axis=1)), f'{opentf.textcolor["red"]}The skill vectors are not multi-hot to show the skill subset!{opentf.textcolor["reset"]}'
    skill_coverages = {int(k): np.zeros(B) for k in topks.split(',')}

    # unit test0: if k grows, this metric increases but for k=inf, it may not reach to exact 1.0 as skill-coverage misses skills in test teams (OOV)
    # unit test1: should make the result max to 1.0 because regardless of the selected expert, any of them, has ALL the skills
    # for i in range(expertskillvecs.shape[0]):
    #     for j in range(expertskillvecs.shape[1]):
    #         expertskillvecs[i, j] = 1  # should make the result max to 1.0

    with tqdm(total=B) as pbar:
        for b in range(B):
            # unit test2: this should make the result nonzero because the required skills are become the entire set, and at least overlaps with an expert's skill
            # for i in range(X[b].shape[1]): X[b, i] = 1

            ranked_experts = np.argsort(Y_[b])[::-1]
            for k in skill_coverages.keys():
                topk_indices = ranked_experts[:k]
                topk_skill_rows = expertskillvecs[topk_indices]  # shape [k, S] sparse
                #from Y_ (recommended experts) to X_ (skills of recommended experts)
                X_ = (topk_skill_rows.max(axis=0) > 0)  # max works as OR, result in [1, S], sparse
                skill_coverages[k][b] = np.dot(X_, X[b].transpose()).sum() / X[b].sum() # let it raise exception if divide by 0 (must not happen as it shows empty required skills)
            pbar.update(1)

    pd = opentf.install_import('pandas==2.0.0', 'pandas')
    df_skc = pd.DataFrame(data=[v for v in skill_coverages.values()], index=[f'skill_coverage_{k}' for k in skill_coverages.keys()])
    return df_skc, df_skc.mean(axis=1).to_frame('mean')

