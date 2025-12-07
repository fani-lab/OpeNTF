import logging, numpy as np
log = logging.getLogger(__name__)

import pkgmgr as opentf
def calculate_metrics(Y, Y_, topK=None, per_instance=False, metrics=['P_2,5', 'recall_2,5', 'ndcg_cut_2,5']):
    pd = opentf.install_import('pandas')
    tqdm = opentf.install_import('tqdm', from_module='tqdm')
    pytrec_eval = opentf.install_import('pytrec-eval-terrier', 'pytrec_eval')
    qrel = dict(); run = dict()
    log.info(f'Building pytrec_eval input for {Y.shape[0]} instances ...')
    k = min(topK, Y_.shape[1]) if topK else Y_.shape[1] #first stage topK for efficiency in space and speed
    with tqdm(total=Y.shape[0]) as pbar:
        topk_idxes = np.argpartition(-Y_, kth=k - 1, axis=1)[:, :k]
        for i in range(Y.shape[0]):
            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in Y[i].nonzero()[1]}
            run['q' + str(i)] = {'d' + str(idx): float(Y_[i][topk_idxes[i][j]]) for j, idx in enumerate(topk_idxes[i])}
            pbar.update(1)
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, set(metrics)).evaluate(run)).transpose()
    df_mean = df.mean().to_frame('mean').rename_axis('metrics')
    return df if per_instance else None, df_mean

def calculate_auc_roc(Y, Y_, curve=False):
    scikit = opentf.install_import('scikit-learn', 'sklearn.metrics')
    auc = scikit.roc_auc_score(Y.toarray(), Y_, average='micro', multi_class="ovr")
    if curve: fpr, tpr, _ = scikit.roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, (fpr, tpr) if curve else None

def calculate_skill_coverage(X, Y_, expertskillvecs, per_instance=False, topks='2,5,10'):#skillcoveragevecs: ExS, X: BatchxS, Y_: BatchxE
    tqdm = opentf.install_import('tqdm', from_module='tqdm')
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

            ranked_experts = np.random.permutation(len(Y_[b])) if np.all(Y_[b] == Y_[b,0]) else np.argsort(Y_[b])[::-1]
            for k in skill_coverages.keys():
                topk_indices = ranked_experts[:k]
                topk_skill_rows = expertskillvecs[topk_indices]  # shape [k, S] sparse
                #from Y_ (recommended experts) to X_ (skills of recommended experts)
                X_ = (topk_skill_rows.max(axis=0) > 0)  # max works as OR, result in [1, S], sparse
                skill_coverages[k][b] = np.dot(X_, X[b].transpose()).sum() / X[b].sum() # let it raise exception if divide by 0 (must not happen as it shows empty required skills)
            pbar.update(1)

    pd = opentf.install_import('pandas')
    df_skc = pd.DataFrame(data=[v for v in skill_coverages.values()], index=[f'skill_coverage_{k}' for k in skill_coverages.keys()]).transpose()
    return df_skc, df_skc.mean().to_frame('mean').rename_axis('metrics')