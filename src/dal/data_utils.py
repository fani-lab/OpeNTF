import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold
import mdl.param
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

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

# def plot_roc(loader, model, device):
#     C = loader.dataset.output.shape[1]
#     with torch.no_grad():
#         y_true = torch.empty(0, C)
#         y_pred = torch.empty(0, C)
#
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#
#             scores = model(x)
#             scores = torch.sigmoid(scores)
#             scores = torch.round(scores)
#
#             y = y.squeeze(1).cpu().numpy()
#             scores = scores.squeeze(1).cpu().numpy()
#
#             y_true = np.vstack((y_true, y))
#             y_pred = np.vstack((y_pred, scores))
#
#         fpr, tpr, _ = roc_curve(y_true, y_pred)
#         plt.plot(fpr, tpr, 'k-')
#
#         # plt.xlabel('Number of nodes')
#         # plt.ylabel('AUC')
#         # plt.title('lr = 0.01')
#         # plt.legend()
#         plt.show()

def roc_auc(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

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
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        f1 = classification_report(y_true, y_pred, zero_division=0)
        return f1


def eval_metrics(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        auc = roc_auc_score(y_true, y_pred, average='micro', multi_class = "ovr")
        cls_rep = classification_report(y_true, y_pred, zero_division=0)
        return str(auc), cls_rep



def prc_auc(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auc = auc(precision, recall)
        return auc


def f_score(loader, model, device, average = 'micro'):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))
        f1 = f1_score(y_true, y_pred, zero_division=0, average=average)
        return f1


def confusion_mat(loader, model, device):
    C = loader.dataset.output.shape[1]
    with torch.no_grad():
        y_true = torch.empty(0, C)
        y_pred = torch.empty(0, C)
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            y_true = np.vstack((y_true, y))
            y_pred = np.vstack((y_pred, scores))

        cm = multilabel_confusion_matrix(y_true, y_pred)
        return cm


def micro_f1(loader, model, device):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            N = y.shape[0]
            C = y.shape[2]
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)

            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()

            for i in range(N):
                for j in range(C):
                    if y[i,j] == 0.0 and scores[i,j] == 0.0:
                        true_neg += 1
                    elif y[i,j] == 1.0 and scores[i,j] == 1.0:
                        true_pos += 1
                    elif y[i,j] == 0.0 and scores[i,j] == 1.0:
                        false_pos += 1
                    elif y[i,j] == 1.0 and scores[i,j] == 0.0:
                        false_neg += 1  

            print("true pos", true_pos)
            print("true neg", true_neg)
            print("false pos", false_pos)
            print("false neg", false_neg)  
                # report all four for each instance
                # zero all four
                # average them
            # accuracy = (scores == y).sum() / (N*C) * 100
            # accuracy = torch.true_divide((scores == y).sum(),(N*C))

            
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1_score = 2 * precision * recall / (precision + recall)
    # model.train()
    return f1_score
    # return accuracy


def special_f1(loader, model, device):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_negs = []
    true_poss = []
    false_poss = []
    false_negs = []
    model.eval()

    with torch.no_grad():
        
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            N = y.shape[0]
            C = y.shape[2]
            
            scores = model(x)
            scores = torch.sigmoid(scores)
            scores = torch.round(scores)
            
            # precisions = []
            # recalls = []
            # f1s = []
            y = y.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            for i in range(N):
                for j in range(C):
                    if y[i,j] == 0.0 and scores[i,j] == 0.0:
                        true_neg += 1
                    elif y[i,j] == 1.0 and scores[i,j] == 1.0:
                        true_pos += 1
                    elif y[i,j] == 0.0 and scores[i,j] == 1.0:
                        false_pos += 1
                    elif y[i,j] == 1.0 and scores[i,j] == 0.0:
                        false_neg += 1   

                true_negs.append(true_neg)
                true_poss.append(true_pos)
                false_poss.append(false_pos)
                false_negs.append(false_neg)

                # precision = true_pos/(true_pos+false_pos)
                # recall = true_pos/(true_pos+false_neg)
                # f1_score = 2 * precision * recall / (precision + recall)

                # precisions.append(precision)
                # recalls.append(recall)
                # f1s.append(f1_score)

                true_neg, true_pos, false_pos, false_neg = 0, 0, 0, 0

        average_tn = sum(true_negs)/len(true_negs)
        average_tp = sum(true_poss)/len(true_poss)
        average_fp = sum(false_poss)/len(false_poss)
        average_fn = sum(false_negs)/len(false_negs)

        # average_precision = sum(precisions)/len(precisions)
        # average_recall = sum(recalls)/len(recalls)
        # average_f1 = sum(f1s)/len(f1s)

        ave_pre = average_tp/(average_tp + average_fp)
        ave_rec = average_tp/(average_tp + average_fn)
        ave_f1 = 2 * ave_pre * ave_rec / (ave_pre + ave_rec)

        print(average_tn)
        print(average_tp)
        print(average_fp)
        print(average_fn)
        # print(average_precision)
        # print(average_recall)
        # print(average_f1)
        print(ave_pre)
        print(ave_rec)
        print(ave_f1)
