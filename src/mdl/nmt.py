import pickle, subprocess, os, re, shlex, numpy as np, logging, sys, scipy
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.ntf import Ntf

class Nmt(Ntf):
    def __init__(self, output, device, seed, cgf):
        super().__init__(output, device, seed, cgf)
        Nmt.onmt = opentf.install_import('OpenNMT-py', 'onmt')  # 3.3 >> it installs its own version of pytorch==2.0.1

    def _prep(self, teamsvecs, splits):
        log.info(f'Loading src and tgt files and/or folding folders for OpenNMT in {self.output} ...')
        if os.path.isfile(f'{self.output}/src-test.txt') and os.path.isfile(f'{self.output}/tgt-test.txt'): return
        log.info(f'Files and/or folders not found! Generating ...')

        input_data = []
        output_data = []
        X = teamsvecs['skill'] if scipy.sparse.issparse(teamsvecs['skill']) else teamsvecs['original_skill']  # if skill dense vectors, fall back to multi-hot
        for i in range(teamsvecs['skill'].shape[0]): #n_teams
            input_data.append([f's{str(skill_idx)}' for skill_idx in X[i].nonzero()[1]])
            output_data.append([f'm{str(member_idx)}' for member_idx in teamsvecs['member'][i].nonzero()[1]])

        input_data = np.array([f'{_}\n' for _ in [' '.join(_) for _ in input_data]])
        output_data = np.array([f'{_}\n' for _ in [' '.join(_) for _ in output_data]])

        for foldidx in splits['folds'].keys():
            with open(f'{self.output}/f{foldidx}.src-train.txt', 'w', newline='') as src_train: src_train.writelines(input_data[splits['folds'][foldidx]['train']])
            with open(f'{self.output}/f{foldidx}.src-valid.txt', 'w', newline='') as src_val: src_val.writelines(input_data[splits['folds'][foldidx]['valid']])
            with open(f'{self.output}/f{foldidx}.tgt-train.txt', 'w', newline='') as tgt_train: tgt_train.writelines(output_data[splits['folds'][foldidx]['train']])
            with open(f'{self.output}/f{foldidx}.tgt-valid.txt', 'w', newline='') as tgt_val: tgt_val.writelines(output_data[splits['folds'][foldidx]['valid']])

            onmtcfg = OmegaConf.load('./mdl/__config__.nmt.yaml')
            if OmegaConf.is_interpolation(onmtcfg, 'seed'): onmtcfg.seed = self.seed
            if OmegaConf.is_interpolation(onmtcfg.data.corpus_1, 'path_src'): onmtcfg.data.corpus_1.path_src = f'{self.output}/f{foldidx}.src-train.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.corpus_1, 'path_tgt'): onmtcfg.data.corpus_1.path_tgt = f'{self.output}/f{foldidx}.tgt-train.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.valid, 'path_src'): onmtcfg.data.valid.path_src = f'{self.output}/f{foldidx}.src-valid.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.valid, 'path_tgt'): onmtcfg.data.valid.path_tgt = f'{self.output}/f{foldidx}.tgt-valid.txt'
            if OmegaConf.is_interpolation(onmtcfg, 'src_vocab'): onmtcfg.src_vocab = f'{self.output}/f{foldidx}.vocab.src'
            if OmegaConf.is_interpolation(onmtcfg, 'tgt_vocab'): onmtcfg.tgt_vocab = f'{self.output}/f{foldidx}.vocab.tgt'
            if OmegaConf.is_interpolation(onmtcfg, 'save_data'): onmtcfg.save_data = f'{self.output}/f{foldidx}.'
            if OmegaConf.is_interpolation(onmtcfg, 'save_model'): onmtcfg.save_model = f'{self.output}/f{foldidx}.'

            log.info(f'{opentf.textcolor["green"]}Overriding onmt.data config for fold{foldidx} in {self.output}/f{foldidx}.config.yml ...{opentf.textcolor["reset"]}')
            OmegaConf.save(onmtcfg, f'{self.output}/f{foldidx}.config.yml', resolve=False)

            # cli_cmd = f'onmt_build_vocab -config {fold_path}.config.yml -n_sample {len(input_data)}'
            # log.info(cli_cmd)
            # p = subprocess.Popen(shlex.split(cli_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # for l in p.stdout: log.info("[onmt_build_vocab stdout] %s", l.strip())
            # for l in p.stderr: log.error("[onmt_build_vocab stderr] %s", l.strip())
            # p.wait()

            #NOTE: it overrides the main command
            onmt_build = opentf.install_import('OpenNMT-py', 'onmt.bin.build_vocab', 'main')
            sys.argv += ['onmt_build_vocab', '-config', f'{self.output}/f{foldidx}.config.yml']
            onmt_build()

        with open(f'{self.output}/src-test.txt', 'w', newline='') as src_test: src_test.writelines(input_data[splits['test']])
        with open(f'{self.output}/tgt-test.txt', 'w', newline='') as tgt_test: tgt_test.writelines(output_data[splits['test']])

    def learn(self, teamsvecs, splits, prev_model):
        self._prep(teamsvecs, splits)
        onmt_train = opentf.install_import('OpenNMT-py', 'onmt.bin.train', 'main')
        for foldidx in splits['folds'].keys():
            train_size = len(splits['folds'][foldidx]['train'])
            onmtcfg = OmegaConf.load(f'{self.output}/f{foldidx}.config.yml')

            if OmegaConf.is_interpolation(onmtcfg, 'world_size'): onmtcfg.world_size = 1 if self.device in ['cpu', 'cuda'] else len(self.device.split(','))
            if OmegaConf.is_interpolation(onmtcfg, 'gpu_ranks'):
                if self.device == 'cpu': onmtcfg.gpu_ranks = []
                elif self.device == 'cuda': onmtcfg.gpu_ranks = [0]
                elif 'cuda:' in self.device: onmtcfg.gpu_ranks = [int(i) for i in self.device.split(':')[1].split(',')]
            if OmegaConf.is_interpolation(onmtcfg, 'save_checkpoint_steps'): onmtcfg.save_checkpoint_steps = int(self.cfg.spe)
            if OmegaConf.is_interpolation(onmtcfg, 'train_steps'): onmtcfg.train_steps = int(np.ceil(train_size / self.cfg.b) * self.cfg.e)
            if OmegaConf.is_interpolation(onmtcfg, 'batch_size'): onmtcfg.batch_size = self.cfg.b
            if OmegaConf.is_interpolation(onmtcfg, 'bucket_size'): onmtcfg.bucket_size = train_size
            if OmegaConf.is_interpolation(onmtcfg, 'learning_rate'): onmtcfg.learning_rate = self.cfg.lr
            if OmegaConf.is_interpolation(onmtcfg, 'early_stopping'): onmtcfg.early_stopping = self.cfg.es
            if OmegaConf.is_interpolation(onmtcfg, 'encoder_type'): onmtcfg.encoder_type = self.cfg.enc
            if OmegaConf.is_interpolation(onmtcfg, 'decoder_type'): onmtcfg.decoder_type = self.cfg.enc
            onmtcfg.num_workers = os.cpu_count() - 1 if onmtcfg.num_workers == -1 else onmtcfg.num_workers
            onmtcfg.pop('n_sample', None) # just for vocab building

            log.info(f'{opentf.textcolor["blue"]}Overriding onmt config for train for fold{foldidx} in {self.output}/f{foldidx}.config.yml ...{opentf.textcolor["reset"]}')
            OmegaConf.save(onmtcfg, f'{self.output}/f{foldidx}.config.yml', resolve=True)

            # cli_cmd = f'onmt_train  -config {self.output}/f{foldidx}/config.yml '
            # log.info(cli_cmd)
            # p = subprocess.Popen(shlex.split(cli_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # for l in p.stdout: log.info("[onmt_train stdout] %s", l.strip())
            # for l in p.stderr: log.error("[onmt_train stderr] %s", l.strip())
            # p.wait()

            #NOTE: it overrides the main command
            sys.argv += ['onmt_train', '-config', f'{self.output}/f{foldidx}.config.yml']
            onmt_train()

    def test(self, teamsvecs, splits, testcfg):
        for foldidx in splits['folds'].keys():
            onmtcfg = OmegaConf.load(f'{self.output}/f{foldidx}.config.yml')
            modelfiles = [f'{self.output}/{_}' for _ in os.listdir(self.output) if re.match(f'f{foldidx}._step_\d+.pt', _)]
            modelfiles = sorted(modelfiles, key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
            if not testcfg.per_epoch: modelfiles = modelfiles[0] #only the last step as the final model if no per_epoch
            for modelfile in modelfiles:
                step = modelfile.split('_')[-1].split('.')[0]
                cli_cmd = 'onmt_translate '
                cli_cmd += f'-model {modelfile} '
                cli_cmd += f'-src {self.output}/src-test.txt '
                cli_cmd += f'-output {self.output}/f{foldidx}.test.e{step}.pred '
                cli_cmd += f'-gpu {self.device.split(":")[1] if ":" in self.device else "0"} ' if 'cuda' in self.device else ''
                cli_cmd += f'--min_length {onmtcfg.min_length} --max_length {onmtcfg.max_length} '
                cli_cmd += f'--beam_size {onmtcfg.beam_size} --n_best {onmtcfg.n_best} '
                cli_cmd += '--replace_unk -verbose '
                print(f'{cli_cmd}')
                p = subprocess.Popen(shlex.split(cli_cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout: log.info("[onmt_translate] %s", line.strip())
                p.wait()

                ## api call version but the cli version is more convinient here.
                # from onmt.translate.translator import build_translator
                # translator = build_translator(opt_path_or_dict="config.yaml",  report_score=True)
                # results = translator.translate(src=["Hello world!", "How are you?"],batch_size=2)
    
    #TODO: many code overlaps with ntf.evaluate ...
    def evaluate(self, teamsvecs, splits, evalcfg):
        pd = opentf.install_import('pandas')
        import evl.metric as metric
        fold_mean = pd.DataFrame()
        mean_std = pd.DataFrame()
        if evalcfg.per_instance: fold_mean_per_instance = pd.DataFrame()
        Y = teamsvecs['member'][splits['test']]
        for foldidx in splits['folds'].keys():
            predfiles = [_ for _ in os.listdir(self.output) if re.match(f'f{foldidx}.test.e\d+.pred$', _)]
            predfiles = sorted(predfiles, key=lambda f: int(re.search(r'\.e(\d+)\.', f).group(1)), reverse=True)
            #per_epoch depends on "save_checkpoint_steps" param in nmt_config.yaml since it simply collect the checkpoint files
            #so, per_epoch => per_checkpoints/steps
            if not evalcfg.per_epoch: predfiles = predfiles[0]  # only the last step as the final model if no per_epoch
            for i, predfile in enumerate(predfiles):
                df_pred = pd.read_csv(f'{self.output}/{predfile}', header=None)
                #df_pred = pd.read_csv(f'{self.output}/tgt-test.txt', header=None) # for unit-test, y_ = y
                #TODO: sparse?
                Y_ = np.zeros((Y.shape[0], Y.shape[1]))
                for _ in range(Y.shape[0]):
                    predlist = (df_pred.iloc[_])[0]
                    predlist = re.sub(r'\baveryunlikelytoken\w*\b', '', predlist).strip().replace('<unk>', '').replace('<s>', '').replace('m', '').split()
                    for pred in predlist: Y_[_, int(pred)] = 1/len(predlist)

                log.info(f'Evaluating predictions at {self.output}/{predfile} ... for {evalcfg.metrics}')

                df, df_mean = pd.DataFrame(), pd.DataFrame()
                if evalcfg.metrics.trec:
                    log.info(f'{evalcfg.metrics.trec} ...')
                    df, df_mean = metric.calculate_metrics(Y, Y_, evalcfg.topK, evalcfg.per_instance, evalcfg.metrics.trec)

                if 'aucroc' in evalcfg.metrics.other:
                    log.info("['aucroc'] and curve values (fpr, tpr) ...")
                    aucroc, fpr_tpr = metric.calculate_auc_roc(Y, Y_)
                    if df_mean.empty: df_mean = pd.DataFrame(columns=['mean'])
                    df_mean.loc['aucroc'] = aucroc
                    with open(f'{self.output}/{predfile}.eval.roc.pkl', 'wb') as outfile: pickle.dump(fpr_tpr, outfile)

                if (m := [m for m in evalcfg.metrics.other if 'skill_coverage' in m]):  # since this metric comes with topks str like 'skill_coverage_2,5,10'
                    log.info(f'{m} ...')
                    X = teamsvecs['skill'] if scipy.sparse.issparse(teamsvecs['skill']) else teamsvecs['original_skill']  # to accomodate dense emb vecs of skills
                    X = X[splits['test']]
                    #TODO: for absolute 0 all, it should be 0?
                    df_skc, df_mean_skc = metric.calculate_skill_coverage(X, Y_, teamsvecs['skillcoverage'], evalcfg.per_instance, topks=m[0].replace('skill_coverage_', ''))
                    if df.empty: df = df_skc
                    else: df_skc.columns = df.columns; df = pd.concat([df, df_skc], axis=0)
                    if df_mean.empty: df_mean = df_mean_skc
                    else: df_mean = pd.concat([df_mean, df_mean_skc], axis=0)

                if evalcfg.per_instance: df.to_csv(f'{self.output}/{predfile}.eval.per_instance.csv', float_format='%.5f')
                log.info(f'Saving file per fold as {self.output}/{predfile}.eval.mean.csv')
                df_mean.to_csv(f'{self.output}/{predfile}.eval.mean.csv')
                if i == 0:  # non-epoch-based only, as there is different number of epochs for each fold model due to earlystopping
                    fold_mean = pd.concat([fold_mean, df_mean], axis=1)
                    if evalcfg.per_instance: fold_mean_per_instance = fold_mean_per_instance.add(df, fill_value=0)
        mean_std['mean'] = fold_mean.mean(axis=1)
        mean_std['std'] = fold_mean.std(axis=1)
        log.info(f'Saving mean evaluation file over {len(splits["folds"])} folds as {self.output}/test.pred.eval.mean.csv')
        mean_std.to_csv(f'{self.output}/test.pred.eval.mean.csv')
        if evalcfg.per_instance: fold_mean_per_instance.truediv(len(splits['folds'].keys())).to_csv(f'{self.output}/test.pred.eval.per_instance_mean.csv')


