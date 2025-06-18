import os, pickle, re, logging, scipy.sparse
from functools import partial
log = logging.getLogger(__name__)

import pkgmgr as opentf
class Ntf:
    def __init__(self, output, pytorch, device, seed, cgf):
        self.cfg = cgf
        self.seed = seed
        self.device = device
        self.model = None
        self.is_bayesian = False
        Ntf.torch = opentf.install_import(pytorch, 'torch')
        opentf.set_seed(self.seed, Ntf.torch)
        opentf.install_import('tensorboard==2.14.0 protobuf==3.20', 'tensorboard')
        self.output = output + '.' + opentf.cfg2str(self.cfg)
        if not os.path.isdir(self.output): os.makedirs(self.output)
        self.writer = opentf.install_import('tensorboardX==2.6.2.2', 'tensorboardX', 'SummaryWriter')
        class NtfDataset(Ntf.torch.utils.data.Dataset):
            def __init__(self, input_matrix, output_matrix):
                super().__init__()
                self.input, self.output = input_matrix, output_matrix
            def __len__(self): return self.input.shape[0]
            def __getitem__(self, index):
                if scipy.sparse.issparse(self.input): return Ntf.torch.as_tensor(self.input[index].toarray()).float(), Ntf.torch.as_tensor(self.output[index].toarray()).float()
                else: return Ntf.torch.as_tensor(self.input[index]).float(), Ntf.torch.as_tensor(self.output[index].toarray()).float()
        Ntf.dataset = NtfDataset

    def name(self): return self.__class__.__name__.lower()

    def learn(self, teamsvecs, indexes, splits, prev_model): pass
    def test(self, teamsvecs, indexes, splits, on_train=False, per_epoch=False): pass
    def evaluate(self, teamsvecs, splits, model_path, on_train=False, per_instance=False, per_epoch=False):
        import pandas as pd
        import torch
        import evl.metric as metric

        log.info(f'\n{opentf.textcolor["blue"]}.............. starting eval .................{opentf.textcolor["reset"]}\n')
        if not os.path.isdir(model_path): raise Exception("The predictions do not exist!")
        y_test = teamsvecs['member'][splits['test']] # the actual y

        for pred_set in (['test', 'train', 'valid'] if on_train else ['test']):
            fold_mean = pd.DataFrame()
            if per_instance: fold_mean_per_instance = pd.DataFrame()
            mean_std = pd.DataFrame()
            
            #there is not such files for random model!!
            predfiles = [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match('state_dict_model.f\d+.pt', _)]
            if per_epoch: predfiles += [f'{model_path}/{_}' for _ in os.listdir(model_path) if re.match('state_dict_model.f\d+.e\d+', _)]

            epochs = len(predfiles)//len(splits['folds'].keys())
            for e in range(epochs):
                epoch = f'e{e}.' if per_epoch and e < (epochs - 1) else ''
                for foldidx in splits['folds'].keys():
                    if pred_set != 'test': Y = teamsvecs['member'][splits['folds'][foldidx][pred_set]]
                    else: Y = y_test
                    Y_ = torch.load(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred')['y_pred']

                    actual_skills = teamsvecs['skill_main'][splits['test']].todense().astype(int) # taking the skills from the test teams
                    skill_coverage = metric.skill_coverage(teamsvecs, actual_skills, Y_, [2, 5, 10]) # dict of skill_coverages for list of k's
                    df_skc = pd.DataFrame.from_dict(skill_coverage, orient='index', columns=['mean']) # skill_coverage (top_k) per fold

                    df, df_mean, (fpr, tpr) = metric.calculate_metrics(Y, Y_, per_instance)
                    if per_instance: df.to_csv(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.per_instance.csv', float_format='%.15f')
                    log.info(f'Saving file per fold as : f{foldidx}.{pred_set}.{epoch}pred.eval.mean.csv')
                    df_mean.to_csv(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.mean.csv')
                    with open(f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred.eval.roc.pkl', 'wb') as outfile: pickle.dump((fpr, tpr), outfile)

                    df_mean = pd.concat([df_mean, df_skc], axis = 0) # concat df_skc to the last row of df_mean
                    fold_mean = pd.concat([fold_mean, df_mean], axis=1)
                    if per_instance: fold_mean_per_instance = fold_mean_per_instance.add(df, fill_value=0)
                # the last row is a list of roc values
                mean_std['mean'] = fold_mean.mean(axis=1)
                mean_std['std'] = fold_mean.std(axis=1)
                log.info(f'Saving mean evaluation file over nfolds as : {pred_set}.{epoch}pred.eval.mean.csv')
                mean_std.to_csv(f'{model_path}/{pred_set}.{epoch}pred.eval.mean.csv')
                if per_instance: fold_mean_per_instance.truediv(len(splits['folds'].keys())).to_csv(f'{model_path}/{pred_set}.{epoch}pred.eval.per_instance_mean.csv')
        log.info(f'\n{opentf.textcolor["blue"]}.............. starting eval .................{opentf.textcolor["reset"]}\n')

    def fair(self, model_path, teamsvecs, splits, settings):
        import multiprocessing
        from Adila.src import main as adila
        if os.path.isfile(model_path):
            adila.Reranking.run(fpred=model_path,
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
                        for fpred, output in pairs: adila.Reranking.run(fpred=fpred,
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
                            executor.starmap(partial(adila.Reranking.run,
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
        import matplotlib.pyplot as plt
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

