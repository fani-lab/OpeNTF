import multiprocessing
import os, pickle, re
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from Adila.src import main
from eval.metric import *

from reranking import ndkl

class Ntf(nn.Module):
    def __init__(self):
        super(Ntf, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def learn(self, splits, indexes, vecs, params, prev_model, output): pass
    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False, merge_skills=False): pass

    def evaluate(self, model_path, splits, vecs, indexes, on_train_valid_set=False, per_instance=False, per_epoch=False):
        if not os.path.isdir(model_path): raise Exception("The predictions do not exist!")
        y_test = vecs['member'][splits['test']]
        f_test = vecs['female'][splits['test']]
        genders = vecs['gender'].nonzero()[1]
        percentage_of_females = len(genders)/y_test.shape[1]
        # desired_distribution = {"f": percentage_of_females, "m": 1-percentage_of_females}
        desired_distribution = {"f": 0.1, "m": 0.9}
        for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
            fold_mean = pd.DataFrame()
            if per_instance: fold_mean_per_instance = pd.DataFrame()
            mean_std = pd.DataFrame()
            
            #there is not such files for random model!!
            predfiles = [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match('state_dict_model.f\d+.pt', _)]
            if per_epoch: predfiles += [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match('state_dict_model.f\d+.e\d+', _)]

            epochs = len(predfiles)//len(splits['folds'].keys())
            for e in range(epochs):
                epoch = f'e{e}.' if per_epoch and e < (epochs - 1) else ""
                for foldidx in splits['folds'].keys():
                    if pred_set != 'test':
                        Y = vecs['member'][splits['folds'][foldidx][pred_set]]
                    else:
                        Y = y_test
                    Y_ = torch.load(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred')

                    topk = 500
                    topk_indices = np.argsort(Y_, axis=1)[:, -topk:]

                    ndkls = []
                    for t in range(Y_.shape[0]):
                        item_attribute = []
                        for i in topk_indices[t]:
                            if i in genders:
                                item_attribute.append("f")
                            else:
                                item_attribute.append("m")
                        ndkls.append(ndkl(item_attribute, desired_distribution))
                    avg_ndkl = sum(ndkls) / len(ndkls)
                    topk_array = np.zeros_like(Y_)

                    # ndkls = []
                    # item_attribute = []
                    # for indices in topk_indices:
                    #     for i in indices:
                    #         if i in genders:
                    #             item_attribute.append("f")
                    #         else:
                    #             item_attribute.append("m")
                    #     ndkls.append(ndkl(item_attribute, desired_distribution))
                    # avg_ndkl = sum(ndkls) / len(ndkls)
                    # topk_array = np.zeros_like(Y_)

                    for i in range(Y_.shape[0]):
                        topk_array[i, topk_indices[i]] = 1

                    female_ids = f_test.toarray()

                    result = female_ids*topk_array

                    female_presence_count = np.sum(result, axis=1)
                    if female_presence_count.ndim == 1:
                        female_presence_count = female_presence_count.reshape((1, -1)).transpose()

                    female_presence_percentage = female_presence_count / topk
                    
                    average_female_percentage_in_all_teams = np.mean(female_presence_percentage)

                    print(f"Count of female indices in top {topk} probabilities for each team in fold {foldidx}:")
                    print(female_presence_count)
                    print(f"Percentage of female indices in top {topk} probabilities for each team in fold {foldidx}:")
                    print(female_presence_percentage)
                    print(f"Average percentage of females in all teams in fold {foldidx}: {average_female_percentage_in_all_teams}")

                    df, df_mean, (fpr, tpr) = calculate_metrics(Y, Y_, per_instance)
                    if per_instance: df.to_csv(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.per_instance.csv', float_format='%.15f')
                    df_mean.loc['avg_female_perc'] = average_female_percentage_in_all_teams
                    df_mean.loc['ndkl'] = avg_ndkl
                    df_mean.to_csv(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.mean.csv')
                    with open(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.roc.pkl', 'wb') as outfile:
                        pickle.dump((fpr, tpr), outfile)
                    fold_mean = pd.concat([fold_mean, df_mean], axis=1)
                    if per_instance: fold_mean_per_instance = fold_mean_per_instance.add(df, fill_value=0)
                # the last row is a list of roc values
                mean_std['mean'] = fold_mean.mean(axis=1)
                mean_std['std'] = fold_mean.std(axis=1)
                mean_std.to_csv(f'{model_path}/{pred_set}.{epoch}pred.eval.mean.csv')
                if per_instance: fold_mean_per_instance.truediv(len(splits['folds'].keys())).to_csv(f'{model_path}/{pred_set}.{epoch}pred.eval.per_instance_mean.csv')

    def fair(self, model_path, teamsvecs, splits, settings):

        if os.path.isfile(model_path):
            main.Reranking.run(fpred=model_path,
                          output=model_path,
                          teamsvecs=teamsvecs,
                          splits=splits,
                          np_ratio=settings['np_ratio'],
                          algorithm=settings['algorithm'],
                          k_max=settings['k_max'],
                          fairness_metrics=settings['fairness_metrics'],
                          eq_op=settings['eq_op'],
                          utility_metrics=settings['utility_metrics'],
                          )
            exit(0)

        if os.path.isdir(model_path):
            # given a root folder, we can crawl the folder to find *.pred files and run the pipeline for all
            files = list()
            for dirpath, dirnames, filenames in os.walk(model_path): files += [
                os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if
                file.endswith("pred") and 'rerank' not in file]

            files = pd.DataFrame(files, columns=['.', '..', 'domain', 'baseline', 'setting', 'setting_',  'rfile'])
            address_list = list()

            pairs = []
            for i, row in files.iterrows():
                output = f"{row['.']}/{row['..']}/{row['domain']}/{row['baseline']}/{row['setting']}/{row['setting_']}/"
                pairs.append((f'{output}{row["rfile"]}', f'{output}rerank/'))

            if settings['mode'] == 0:  # sequential run
                for algorithm in settings['fairness']:
                    for att in settings['attribute']:
                        for fpred, output in pairs: main.Reranking.run(fpred=fpred,
                                                                  output=output,
                                                                  teamsvecs=teamsvecs,
                                                                  splits=splits,
                                                                  np_ratio=settings['np_ratio'],
                                                                  algorithm=algorithm,
                                                                  k_max=settings['k_max'],
                                                                  fairness_metrics=settings['fairness_metrics'],
                                                                  eq_op=settings['eq_op'],
                                                                  utility_metrics=settings['utility_metrics'],
                                                                  att=att)
            elif settings['mode'] == 1:  # parallel run
                print(f'Parallel run started ...')
                for algorithm in settings['fairness']:
                    for att in settings['attribute']:
                        with multiprocessing.Pool(multiprocessing.cpu_count() if settings['core'] < 0 else settings['core']) as executor:
                            executor.starmap(partial(main.Reranking.run,
                                                     teamsvecs=teamsvecs,
                                                     splits=splits,
                                                     np_ratio=settings['np_ratio'],
                                                     algorithm=algorithm,
                                                     k_max=settings['k_max'],
                                                     fairness_metrics=settings['fairness_metrics'],
                                                     utility_metrics=settings['utility_metrics'],
                                                     eq_op=settings['eq_op'],
                                                     att=att), pairs)
    def plot_roc(self, model_path, splits, on_train_valid_set=False):
        for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
            plt.figure()
            for foldidx in splits['folds'].keys():
                with open(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'rb') as infile: (fpr, tpr) = pickle.load(infile)
                # fpr, tpr = eval(pd.read_csv(f'{model_path}/f{foldidx}.{pred_set}.pred.eval.mean.csv', index_col=0).loc['roc'][0].replace('array', 'np.array'))
                plt.plot(fpr, tpr, label=f'micro-average fold{foldidx} on {pred_set} set', linestyle=':', linewidth=4)

            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f'ROC curves for {pred_set} set')
            plt.legend()
            plt.savefig(f'{model_path}/{pred_set}.roc.png', dpi=100, bbox_inches='tight')
            plt.show()

    def run(self, splits, vecs, indexes, output, settings, cmd, fair_settings, merge_skills):
        # output = f"{output}/t{vecs['skill'].shape[0]}.s{vecs['skill'].shape[1]}.m{vecs['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings.items() if v])}"
        if not os.path.isdir(output): os.makedirs(output)

        on_train_valid_set = False #random baseline cannot join this.
        per_instance = True
        per_epoch = False

        if 'train' in cmd: self.learn(splits, indexes, vecs, settings, None, output)
        if 'test' in cmd: self.test(output, splits, indexes, vecs, settings, on_train_valid_set, per_epoch, merge_skills)
        if 'eval' in cmd: self.evaluate(output, splits, vecs, indexes, on_train_valid_set, per_instance, per_epoch)
        if 'plot' in cmd: self.plot_roc(output, splits, on_train_valid_set)
        if 'fair' in cmd: self.fair(output, vecs, splits, fair_settings)
