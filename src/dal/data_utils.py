import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, average_precision_score, ndcg_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pytrec_eval
import json

def create_evaluation_splits(n_sample, n_folds):
    train, test = train_test_split(np.arange(n_sample), train_size=0.85, test_size=0.15, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        splits['folds'][k]['valid'] = train[validIdx]

    return splits

def evaluation_metrics_mini(loader, model, device):
    C = loader.dataset.output.shape[1]
    eval_met = {'p': "", 'r': "", 'ndcg': "", 'map': "", 'auc': ""}
    with torch.no_grad():
        torch.cuda.empty_cache()
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        counter = 0
        aucs = []
        fprs, tprs = [], []
        mean_p_at_ks, p_keyss, p_valuess = [], [], []
        mean_r_at_ks, r_keyss, r_valuess = [], [], []
        mean_ndcg_at_ks, ndcg_keyss, ndcg_valuess = [], [], []
        mean_map_at_ks, map_keyss, map_valuess = [], [], []
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))
            print(counter)
            counter += 1

            auc_ = roc_auc_score(y_true, y_pred, average='micro', multi_class="ovr")
            aucs.append(auc_)
            # fpr_, tpr_, _ = roc_curve(y_true.ravel(), y_pred.ravel())
            # fprs.append(fpr_)
            # tprs.append(tpr_)
            # eval_met['auc'] = auc_
            print(f"AUC is: {str(auc_)}")
            # eval_met['roc'] = (fpr_, tpr_)
            # print("ROC is measured.")

            true_indices = np.transpose(np.nonzero(y_true))
            predicted_indices = np.transpose(np.where(y_pred > 0.5))
            for i in range(true_indices.shape[0]):
                if ('q' + str(true_indices[i][0])) not in qrel:
                    qrel['q' + str(true_indices[i][0])] = dict()
                qrel['q' + str(true_indices[i][0])]['d' + str(true_indices[i][1])] = 1

            for i in range(predicted_indices.shape[0]):
                if ('q' + str(predicted_indices[i][0])) not in run:
                    run['q' + str(predicted_indices[i][0])] = dict()
                run['q' + str(predicted_indices[i][0])]['d' + str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]] * 10))
            print("Qrel and run ready for pytrec.")

            # Precision at k
            p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_2,4,6,8,10'})
            # print(json.dumps(evaluator.evaluate(run), indent=1))
            p_at_k = p_evaluator.evaluate(run)
            sum_p_at_k = dict()
            mean_p_at_k = dict()
            for metrics in list(p_at_k.values()):
                for k, v in metrics.items():
                    if k not in sum_p_at_k:
                        sum_p_at_k[k] = v
                    else:
                        sum_p_at_k[k] += v
            for k, v in sum_p_at_k.items():
                mean_p_at_k[k] = sum_p_at_k[k] / len(p_at_k)
            p_keys = [int(k.replace("P_", "")) for k in mean_p_at_k.keys()]
            p_values = list(mean_p_at_k.values())
            mean_p_at_ks.append(mean_p_at_k)
            p_keyss.append(p_keys)
            p_valuess.append(p_values)
            # eval_met['p'] = (mean_p_at_k, p_keys, p_values)
            print("Precision at k is measured.")

            # Recall at k
            r_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall_2,4,6,8,10'})
            # print(json.dumps(evaluator.evaluate(run), indent=1))
            r_at_k = r_evaluator.evaluate(run)
            sum_r_at_k = dict()
            mean_r_at_k = dict()
            for metrics in list(r_at_k.values()):
                for k, v in metrics.items():
                    if k not in sum_r_at_k:
                        sum_r_at_k[k] = v
                    else:
                        sum_r_at_k[k] += v
            for k, v in sum_r_at_k.items():
                mean_r_at_k[k] = sum_r_at_k[k] / len(r_at_k)
            r_keys = [int(k.replace("recall_", "")) for k in mean_r_at_k.keys()]
            r_values = list(mean_r_at_k.values())
            mean_r_at_ks.append(mean_r_at_k)
            r_keyss.append(r_keys)
            r_valuess.append(r_values)
            # eval_met['r'] = (mean_r_at_k, r_keys, r_values)
            print("Recall at k is measured.")

            # ndcg at k
            ndcg_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_2,4,6,8,10'})
            # print(json.dumps(evaluator.evaluate(run), indent=1))
            ndcg_at_k = ndcg_evaluator.evaluate(run)
            sum_ndcg_at_k = dict()
            mean_ndcg_at_k = dict()
            for metrics in list(ndcg_at_k.values()):
                for k, v in metrics.items():
                    if k not in sum_ndcg_at_k:
                        sum_ndcg_at_k[k] = v
                    else:
                        sum_ndcg_at_k[k] += v
            for k, v in sum_ndcg_at_k.items():
                mean_ndcg_at_k[k] = sum_ndcg_at_k[k] / len(ndcg_at_k)

            n_keys = [int(k.replace("ndcg_cut_", "")) for k in mean_ndcg_at_k.keys()]
            n_values = list(mean_ndcg_at_k.values())
            mean_ndcg_at_ks.append(mean_ndcg_at_k)
            ndcg_keyss.append(n_keys)
            ndcg_valuess.append(n_values)
            # eval_met['ndcg'] = (mean_ndcg_at_k, n_keys, n_values)
            print("NDCG at k is measured.")

            # map at k
            map_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_2,4,6,8,10'})
            # print(json.dumps(evaluator.evaluate(run), indent=1))
            map_at_k = map_evaluator.evaluate(run)
            sum_map_at_k = dict()
            mean_map_at_k = dict()
            for metrics in list(map_at_k.values()):
                for k, v in metrics.items():
                    if k not in sum_map_at_k:
                        sum_map_at_k[k] = v
                    else:
                        sum_map_at_k[k] += v
            for k, v in sum_map_at_k.items():
                mean_map_at_k[k] = sum_map_at_k[k] / len(map_at_k)
            m_keys = [int(k.replace("map_cut_", "")) for k in mean_map_at_k.keys()]
            m_values = list(mean_map_at_k.values())
            mean_map_at_ks.append(mean_map_at_k)
            map_keyss.append(m_keys)
            map_valuess.append(m_values)
            # eval_met['map'] = (mean_map_at_k, m_keys, m_values)
            print("MAP at k is measured.")

        auc = sum(aucs)/len(aucs)
        print("AUCs:\n", auc)
        # print("TPRs:\n", [sum(x)/len(tprs) for x in zip(*tprs)])
        # print("FPRs:\n", [sum(x)/len(fprs) for x in zip(*fprs)])

        p_keys = p_keyss[0]
        print("p_keys:\n", p_keys)
        p_values = [sum(x)/len(p_valuess) for x in zip(*p_valuess)]
        print("p_values:\n", p_values)
        mean_p_at_k = {p_keys[i]: p_values[i] for i in range(len(p_keys))}
        print(mean_p_at_k)

        r_keys = r_keyss[0]
        print("r_keys:\n", r_keys)
        r_values = [sum(x)/len(r_valuess) for x in zip(*r_valuess)]
        print("r_values:\n", r_values)
        mean_r_at_k = {r_keys[i]: r_values[i] for i in range(len(r_keys))}
        print("mean_rs:\n", mean_r_at_k)

        ndcg_keys = ndcg_keyss[0]
        print("n_keys:\n", ndcg_keys)
        ndcg_values = [sum(x)/len(ndcg_valuess) for x in zip(*ndcg_valuess)]
        print("n_values:\n", ndcg_values)
        mean_ndcg_at_k = {ndcg_keys[i]: ndcg_values[i] for i in range(len(ndcg_keys))}
        print("mean_ns:\n", mean_ndcg_at_k)

        map_keys = map_keyss[0]
        print("m_keys:\n", map_keys)
        map_values = [sum(x)/len(map_valuess) for x in zip(*map_valuess)]
        print("m_values:\n", map_values)
        mean_map_at_k = {map_keys[i]: map_values[i] for i in range(len(map_keys))}
        print("mean_ms:\n", mean_map_at_k)

        eval_met['auc'] = auc
        eval_met['p'] = (mean_p_at_k, p_keys, p_values)
        eval_met['r'] = (mean_r_at_k, r_keys, r_values)
        eval_met['ndcg'] = (mean_ndcg_at_k, ndcg_keys, ndcg_values)
        eval_met['map'] = (mean_map_at_k, map_keys, map_values)

        return eval_met

def evaluation_metrics(loader, model, device):
    C = loader.dataset.output.shape[1]
    eval_met = {'p': "", 'r': "", 'ndcg': "", 'map': "", 'auc': "", 'roc': ""}
    print(eval_met)
    torch.cuda.empty_cache()
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        counter = 0
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))
            print(counter)
            counter += 1

        auc = str(roc_auc_score(y_true, y_pred, average='micro', multi_class="ovr"))
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        eval_met['auc'] = auc
        print(f"AUC is: {auc}")
        eval_met['roc'] = (fpr, tpr)
        print("ROC is measured.")

        true_indices = np.transpose(np.nonzero(y_true))
        predicted_indices = np.transpose(np.where(y_pred > 0.5))
        for i in range(true_indices.shape[0]):
            if ('q' + str(true_indices[i][0])) not in qrel:
                qrel['q' + str(true_indices[i][0])] = dict()
            qrel['q' + str(true_indices[i][0])]['d' + str(true_indices[i][1])] = 1

        for i in range(predicted_indices.shape[0]):
            if ('q' + str(predicted_indices[i][0])) not in run:
                run['q' + str(predicted_indices[i][0])] = dict()
            run['q' + str(predicted_indices[i][0])]['d' + str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]] * 10))
        print("Qrel and run ready for pytrec.")

        # Precision at k
        p_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        p_at_k = p_evaluator.evaluate(run)
        sum_p_at_k = dict()
        mean_p_at_k = dict()
        for metrics in list(p_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_p_at_k:
                    sum_p_at_k[k] = v
                else:
                    sum_p_at_k[k] += v
        for k, v in sum_p_at_k.items():
            mean_p_at_k[k] = sum_p_at_k[k] / len(p_at_k)
        p_keys = [int(k.replace("P_", "")) for k in mean_p_at_k.keys()]
        p_values = list(mean_p_at_k.values())
        eval_met['p'] = (mean_p_at_k, p_keys, p_values)
        print("Precision at k is measured.")

        # Recall at k
        r_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        r_at_k = r_evaluator.evaluate(run)
        sum_r_at_k = dict()
        mean_r_at_k = dict()
        for metrics in list(r_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_r_at_k:
                    sum_r_at_k[k] = v
                else:
                    sum_r_at_k[k] += v
        for k, v in sum_r_at_k.items():
            mean_r_at_k[k] = sum_r_at_k[k] / len(r_at_k)
        r_keys = [int(k.replace("recall_", "")) for k in mean_r_at_k.keys()]
        r_values = list(mean_r_at_k.values())
        eval_met['r'] = (mean_r_at_k, r_keys, r_values)
        print("Recall at k is measured.")

        # ndcg at k
        ndcg_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        ndcg_at_k = ndcg_evaluator.evaluate(run)
        sum_ndcg_at_k = dict()
        mean_ndcg_at_k = dict()
        for metrics in list(ndcg_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_ndcg_at_k:
                    sum_ndcg_at_k[k] = v
                else:
                    sum_ndcg_at_k[k] += v
        for k, v in sum_ndcg_at_k.items():
            mean_ndcg_at_k[k] = sum_ndcg_at_k[k] / len(ndcg_at_k)

        n_keys = [int(k.replace("ndcg_cut_", "")) for k in mean_ndcg_at_k.keys()]
        n_values = list(mean_ndcg_at_k.values())
        eval_met['ndcg'] = (mean_ndcg_at_k, n_keys, n_values)
        print("NDCG at k is measured.")

        # map at k
        map_evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        map_at_k = map_evaluator.evaluate(run)
        sum_map_at_k = dict()
        mean_map_at_k = dict()
        for metrics in list(map_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_map_at_k:
                    sum_map_at_k[k] = v
                else:
                    sum_map_at_k[k] += v
        for k, v in sum_map_at_k.items():
            mean_map_at_k[k] = sum_map_at_k[k] / len(map_at_k)
        m_keys = [int(k.replace("map_cut_", "")) for k in mean_map_at_k.keys()]
        m_values = list(mean_map_at_k.values())
        eval_met['map'] = (mean_map_at_k, m_keys, m_values)
        print("MAP at k is measured.")
        return eval_met

def precision_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        counter = 0
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))
            print(counter)
            counter += 1

        true_indices = np.transpose(np.nonzero(y_true))
        predicted_indices = np.transpose(np.where(y_pred > 0.5))
        for i in range(true_indices.shape[0]):
            if ('q' + str(true_indices[i][0])) not in qrel:
                qrel['q' + str(true_indices[i][0])] = dict()
            qrel['q' + str(true_indices[i][0])]['d'+str(true_indices[i][1])] = 1

        for i in range(predicted_indices.shape[0]):
            if ('q' + str(predicted_indices[i][0])) not in run:
                run['q' + str(predicted_indices[i][0])] = dict()
            run['q' + str(predicted_indices[i][0])]['d'+str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]]*10))

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        p_at_k = evaluator.evaluate(run)
        sum_p_at_k = dict()
        mean_p_at_k = dict()
        for metrics in list(p_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_p_at_k:
                    sum_p_at_k[k] = v
                else:
                    sum_p_at_k[k] += v
        for k, v in sum_p_at_k.items():
            mean_p_at_k[k] = sum_p_at_k[k]/len(p_at_k)
        p_keys = [int(k.replace("P_", "")) for k in mean_p_at_k.keys()]
        p_values = list(mean_p_at_k.values())
        return mean_p_at_k, p_keys, p_values

def recall_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        true_indices = np.transpose(np.nonzero(y_true))
        predicted_indices = np.transpose(np.where(y_pred > 0.5))
        for i in range(true_indices.shape[0]):
            if ('q' + str(true_indices[i][0])) not in qrel:
                qrel['q' + str(true_indices[i][0])] = dict()
            qrel['q' + str(true_indices[i][0])]['d'+str(true_indices[i][1])] = 1

        for i in range(predicted_indices.shape[0]):
            if ('q' + str(predicted_indices[i][0])) not in run:
                run['q' + str(predicted_indices[i][0])] = dict()
            run['q' + str(predicted_indices[i][0])]['d'+str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]]*10))

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        r_at_k = evaluator.evaluate(run)
        sum_r_at_k = dict()
        mean_r_at_k = dict()
        for metrics in list(r_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_r_at_k:
                    sum_r_at_k[k] = v
                else:
                    sum_r_at_k[k] += v
        for k, v in sum_r_at_k.items():
            mean_r_at_k[k] = sum_r_at_k[k]/len(r_at_k)
        r_keys = [int(k.replace("recall_", "")) for k in mean_r_at_k.keys()]
        r_values = list(mean_r_at_k.values())
        return mean_r_at_k, r_keys, r_values

def ndcg_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        true_indices = np.transpose(np.nonzero(y_true))
        predicted_indices = np.transpose(np.where(y_pred > 0.5))
        for i in range(true_indices.shape[0]):
            if ('q' + str(true_indices[i][0])) not in qrel:
                qrel['q' + str(true_indices[i][0])] = dict()
            qrel['q' + str(true_indices[i][0])]['d'+str(true_indices[i][1])] = 1

        for i in range(predicted_indices.shape[0]):
            if ('q' + str(predicted_indices[i][0])) not in run:
                run['q' + str(predicted_indices[i][0])] = dict()
            run['q' + str(predicted_indices[i][0])]['d'+str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]]*10))

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        ndcg_at_k = evaluator.evaluate(run)
        sum_ndcg_at_k = dict()
        mean_ndcg_at_k = dict()
        for metrics in list(ndcg_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_ndcg_at_k:
                    sum_ndcg_at_k[k] = v
                else:
                    sum_ndcg_at_k[k] += v
        for k, v in sum_ndcg_at_k.items():
            mean_ndcg_at_k[k] = sum_ndcg_at_k[k]/len(ndcg_at_k)

        n_keys = [int(k.replace("ndcg_cut_", "")) for k in mean_ndcg_at_k.keys()]
        n_values = list(mean_ndcg_at_k.values())
        return mean_ndcg_at_k, n_keys, n_values

def map_at_k(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        qrel = dict()
        run = dict()
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        true_indices = np.transpose(np.nonzero(y_true))
        predicted_indices = np.transpose(np.where(y_pred > 0.5))
        for i in range(true_indices.shape[0]):
            if ('q' + str(true_indices[i][0])) not in qrel:
                qrel['q' + str(true_indices[i][0])] = dict()
            qrel['q' + str(true_indices[i][0])]['d'+str(true_indices[i][1])] = 1

        for i in range(predicted_indices.shape[0]):
            if ('q' + str(predicted_indices[i][0])) not in run:
                run['q' + str(predicted_indices[i][0])] = dict()
            run['q' + str(predicted_indices[i][0])]['d'+str(predicted_indices[i][1])] = int(np.round(y_pred[predicted_indices[i][0]][predicted_indices[i][1]]*10))

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_2,4,6,8,10'})
        # print(json.dumps(evaluator.evaluate(run), indent=1))
        map_at_k = evaluator.evaluate(run)
        sum_map_at_k = dict()
        mean_map_at_k = dict()
        for metrics in list(map_at_k.values()):
            for k, v in metrics.items():
                if k not in sum_map_at_k:
                    sum_map_at_k[k] = v
                else:
                    sum_map_at_k[k] += v
        for k, v in sum_map_at_k.items():
            mean_map_at_k[k] = sum_map_at_k[k]/len(map_at_k)
        m_keys = [int(k.replace("map_cut_", "")) for k in mean_map_at_k.keys()]
        m_values = list(mean_map_at_k.values())
        return mean_map_at_k, m_keys, m_values

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
