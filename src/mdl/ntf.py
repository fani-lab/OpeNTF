import os, pickle, re, logging, scipy.sparse
log = logging.getLogger(__name__)

import pkgmgr as opentf
class Ntf:
    def __init__(self, output, device, seed, cfg):
        self.cfg = cfg
        self.seed = seed
        self.device = device
        self.model = None
        self.is_bayesian = False
        Ntf.torch = opentf.install_import('torch')
        self.writer = opentf.install_import(pkg_name='tensorboardX', from_module='SummaryWriter')
        opentf.set_seed(self.seed, Ntf.torch)
        self.output = output + self.name()
        if not os.path.isdir(self.output): os.makedirs(self.output)
        class NtfDataset(Ntf.torch.utils.data.Dataset):
            def __init__(self, input_matrix, output_matrix):
                super().__init__()
                self.input, self.output = input_matrix, output_matrix
            def __len__(self): return self.input.shape[0]
            def __getitem__(self, index):
                if scipy.sparse.issparse(self.input): return Ntf.torch.as_tensor(self.input[index].tocsr().toarray()).float(), Ntf.torch.as_tensor(self.output[index].tocsr().toarray()).float()
                else: return Ntf.torch.as_tensor(self.input[index]).float(), Ntf.torch.as_tensor(self.output[index].tocsr().toarray()).float()
        Ntf.dataset = NtfDataset

    def name(self): return f'/{self.__class__.__name__.lower()}.{opentf.cfg2str(self.cfg)}'

    def learn(self, teamsvecs, splits, prev_model): pass
    def test(self, teamsvecs, splits, testcfg): pass

    #TODO: many code overlaps for nmt.evaluate ... for now, changes here should also apply in nmt.evaluate()
    def evaluate(self, teamsvecs, splits, evalcfg):
        assert os.path.isdir(self.output), f'{opentf.textcolor["red"]}No folder for {self.output} exist!{opentf.textcolor["reset"]}'
        pd = opentf.install_import('pandas')
        import evl.metric as metric
        y_test = teamsvecs['member'][splits['test']]
        # Rnd model does only have f*.test.pred files, no train or valid files >> skip them
        for pred_set in (['test', 'train', 'valid'] if evalcfg.on_train else ['test']):
            fold_mean = pd.DataFrame()
            mean_std = pd.DataFrame()
            if evalcfg.per_instance: fold_mean_per_instance = pd.DataFrame()
            
            for foldidx in splits['folds'].keys(): #for e in range(epochs):
                if pred_set != 'test': Y = teamsvecs['member'][splits['folds'][foldidx][pred_set]]
                else: Y = y_test

                predfiles = [f'{self.output}/f{foldidx}.{pred_set}.pred'] #the first file as a hook
                if evalcfg.per_epoch: predfiles += [f'{self.output}/{_}' for _ in os.listdir(self.output) if re.match(f'f{foldidx}.{pred_set}.e\d+.pred$', _)]
                for i, predfile in enumerate(sorted(sorted(predfiles), key=len)): #the first file is/should be non-epoch-based
                    # in test, we accumulate preds in main memory when ntf.test() due to issue #320: https://github.com/fani-lab/OpeNTF/issues/320
                    # so all preds tensors are already in cpu, but just in case map_location='cpu'
                    Y_ = Ntf.torch.load(predfile, map_location='cpu')['y_pred']
                    log.info(f'Evaluating predictions at {predfile} ... for {evalcfg.metrics}')

                    #evl.metric works on numpy or scipy.sparse. so, we need to convert Y_ which is torch.tensor, either sparse or not
                    Y_ = Y_.to_dense().numpy()

                    df, df_mean = pd.DataFrame(), pd.DataFrame()
                    if evalcfg.metrics.trec:
                        log.info(f'{evalcfg.metrics.trec} ...')
                        df, df_mean = metric.calculate_metrics(Y, Y_, evalcfg.topK, evalcfg.per_instance, evalcfg.metrics.trec)

                    if (m:=[m for m in evalcfg.metrics.other if 'aucroc' in m]):
                        log.info(f'{m} ...')
                        aucroc, fpr_tpr = metric.calculate_auc_roc(Y, Y_, curve=True) if m[0] == 'aucroc+' else metric.calculate_auc_roc(Y, Y_)
                        if df_mean.empty: df_mean = pd.DataFrame(columns=['mean'])
                        df_mean.loc['aucroc'] = aucroc
                        if fpr_tpr:
                            with open(f'{predfile}.eval.roc.pkl', 'wb') as outfile: pickle.dump(fpr_tpr, outfile)

                    if (m:=[m for m in evalcfg.metrics.other if 'skill_coverage' in m]): #since this metric comes with topks str like 'skill_coverage_2,5,10'
                        log.info(f'{m} ...')
                        X = teamsvecs['skill'] if scipy.sparse.issparse(teamsvecs['skill']) else teamsvecs['original_skill'] #to accomodate dense emb vecs of skills
                        X = X[splits['folds'][foldidx][pred_set]] if pred_set != 'test' else X[splits['test']]
                        df_skc, df_mean_skc = metric.calculate_skill_coverage(X, Y_, teamsvecs['skillcoverage'], evalcfg.per_instance, topks=m[0].replace('skill_coverage_', ''))
                        if df.empty: df = df_skc
                        else: df_skc.columns = df.columns; df = pd.concat([df, df_skc], axis=0)
                        if df_mean.empty: df_mean = df_mean_skc
                        else: df_mean = pd.concat([df_mean, df_mean_skc], axis=0)

                    if evalcfg.per_instance: df.to_csv(f'{predfile}.eval.per_instance.csv', float_format='%.5f')
                    log.info(f'Saving file per fold as {predfile}.eval.mean.csv')
                    df_mean.to_csv(f'{predfile}.eval.mean.csv')
                    if i == 0: # non-epoch-based only, as there is different number of epochs for each fold model due to earlystopping
                        fold_mean = pd.concat([fold_mean, df_mean], axis=1)
                        if evalcfg.per_instance: fold_mean_per_instance = fold_mean_per_instance.add(df, fill_value=0)
            mean_std['mean'] = fold_mean.mean(axis=1)
            mean_std['std'] = fold_mean.std(axis=1)
            log.info(f'Saving mean evaluation file over {len(splits["folds"])} folds as {self.output}/{pred_set}.pred.eval.mean.csv')
            mean_std.to_csv(f'{self.output}/{pred_set}.pred.eval.mean.csv')
            if evalcfg.per_instance: fold_mean_per_instance.truediv(len(splits['folds'].keys())).to_csv(f'{self.output}/{pred_set}.pred.eval.per_instance_mean.csv')
    def plot_roc(self, splits, on_train=False):
        plt = opentf.install_import('matplotlib')
        for pred_set in (['test', 'train', 'valid'] if on_train else ['test']):
            plt.figure()
            for foldidx in splits['folds'].keys():
                with open(f'{self.output}/f{foldidx}.{pred_set}.pred.eval.roc.pkl', 'rb') as infile: (fpr, tpr) = pickle.load(infile)
                plt.plot(fpr, tpr, label=f'micro-average fold{foldidx} on {pred_set} set', linestyle=':', linewidth=4)

            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f'ROC curves for {pred_set} set')
            plt.legend()
            plt.savefig(f'{self.output}/{pred_set}.roc.png', dpi=100, bbox_inches='tight')
            # plt.show()

    def fair(self, model_path, teamsvecs, splits, settings):
        import multiprocessing
        from functools import partial
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

            files = pd.DataFrame(files, columns=['.', '..', 'domain', 'baseline', 'setting', 'setting_', 'rfile'])
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
                        with multiprocessing.Pool(
                                multiprocessing.cpu_count() if settings['core'] < 0 else settings['core']) as executor:
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
