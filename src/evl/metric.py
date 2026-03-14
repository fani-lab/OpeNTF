import logging, numpy as np, scipy.sparse as sp
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
        for i in range(Y.shape[0]):
            if sp.issparse(Y_):
                assert sp.isspmatrix_csr(Y_), f'{opentf.textcolor["red"]}For sparse pred files, it must be in csr!{opentf.textcolor["reset"]}'
                start, end = Y_.indptr[i], Y_.indptr[i + 1]
                cols, vals = Y_.indices[start:end], Y_.data[start:end]
                if len(vals) > k:
                    idx = np.argpartition(-vals, k - 1)[:k]
                    idx = idx[np.argsort(-vals[idx])]
                else: idx = np.argsort(-vals)
                top_cols, top_vals = cols[idx], vals[idx]
            else:
                row = Y_[i]
                idx = np.argpartition(-row, k - 1)[:k]
                idx = idx[np.argsort(-row[idx])]
                top_cols, top_vals = idx, row[idx]

            qrel['q' + str(i)] = {'d' + str(idx): 1 for idx in Y[i].nonzero()[1]}
            run['q' + str(i)] = {'d' + str(c): float(v) for c, v in zip(top_cols, top_vals)}
            pbar.update(1)
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, set(metrics)).evaluate(run)).transpose()
    df_mean = df.mean().to_frame('mean').rename_axis('metrics')
    return df if per_instance else None, df_mean

def calculate_auc_roc(Y, Y_, curve=False):
    scikit = opentf.install_import('scikit-learn', 'sklearn.metrics')
    auc = scikit.roc_auc_score(Y.toarray(), Y_.toarray() if sp.issparse(Y_) else Y_, average='micro', multi_class='ovr')
    if curve: fpr, tpr, _ = scikit.roc_curve(Y.toarray().ravel(), Y_.ravel())
    return auc, (fpr, tpr) if curve else None

def calculate_skill_coverage(X, Y_, expertskillvecs, per_instance=False, topks='2,5,10'):#skillcoveragevecs: ExS, X: BatchxS, Y_: BatchxE
    tqdm = opentf.install_import('tqdm', from_module='tqdm')
    pd = opentf.install_import('pandas')

    teams, experts = Y_.shape #batches of output expert recommendations for each team
    assert not np.any(np.all(X.toarray() != 0, axis=1)), f'{opentf.textcolor["red"]}The skill vectors are not multi-hot to show the skill subset!{opentf.textcolor["reset"]}'
    skill_coverages = {int(k): np.zeros(teams) for k in topks.split(',')}

    # unit test0: if k grows, this metric increases but for k=inf, it may not reach to exact 1.0 as skill-coverage misses skills in test teams (OOV)
    # unit test1: should make the result max to 1.0 because regardless of the selected expert, any of them, has ALL the skills
    # for i in range(expertskillvecs.shape[0]):
    #     for j in range(expertskillvecs.shape[1]):
    #         expertskillvecs[i, j] = 1  # should make the result max to 1.0

    with tqdm(total=teams) as pbar:
        for t in range(teams):
            # unit test2: this should make the result nonzero because the required skills are the entire set, and at least overlaps with an expert's skill
            # for i in range(X[t].shape[1]): X[t, i] = 1
            ranked_experts = np.argsort(Y_.getrow(t).toarray().ravel())[::-1] if sp.issparse(Y_) else np.argsort(Y_[t])[::-1]
            for k in skill_coverages.keys():
                topk_indices = ranked_experts[:k]
                topk_skill_rows = expertskillvecs[topk_indices]  # shape [k, S] sparse
                #from Y_ (recommended experts) to X_ (skills of recommended experts)
                X_ = (topk_skill_rows.max(axis=0) > 0)  # max works as OR, result in [1, S], sparse
                skill_coverages[k][t] = np.dot(X_, X[t].transpose()).sum() / X[t].sum() # let it raise exception if divide by 0 (must not happen as it shows empty required skills)
            pbar.update(1)

    df_skc = pd.DataFrame(data=[v for v in skill_coverages.values()], index=[f'skill_coverage_{k}' for k in skill_coverages.keys()]).transpose()
    return df_skc, df_skc.mean().to_frame('mean').rename_axis('metrics')