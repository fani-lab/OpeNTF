import pickle, subprocess, os, re, shlex, numpy as np, logging, sys, scipy
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.ntf import Ntf

class Nmt(Ntf):
    def __init__(self, output, pytorch, device, seed, cgf):
        Nmt.onmt = opentf.install_import('OpenNMT-py==3.3', 'onmt') #3.3 >> it installs its own version of pytorch==2.0.1
        super().__init__(output, None, device, seed, cgf)

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

        endl = '\n'
        input_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in input_data]])
        output_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in output_data]])

        for foldidx in splits['folds'].keys():
            fold_path = f'{self.output}/f{foldidx}'
            if not os.path.isdir(fold_path): os.makedirs(fold_path)

            with open(f'{fold_path}/src-train.txt', 'w') as src_train: src_train.writelines(input_data[splits['folds'][foldidx]['train']])
            with open(f'{fold_path}/src-valid.txt', 'w') as src_val: src_val.writelines(input_data[splits['folds'][foldidx]['valid']])
            with open(f'{fold_path}/tgt-train.txt', 'w') as tgt_train: tgt_train.writelines(output_data[splits['folds'][foldidx]['train']])
            with open(f'{fold_path}/tgt-valid.txt', 'w') as tgt_val: tgt_val.writelines(output_data[splits['folds'][foldidx]['valid']])

            onmtcfg = OmegaConf.load('./mdl/__config__.nmt.yaml')
            if OmegaConf.is_interpolation(onmtcfg, 'seed'): onmtcfg.seed = self.seed
            if OmegaConf.is_interpolation(onmtcfg.data.corpus_1, 'path_src'): onmtcfg.data.corpus_1.path_src = f'{fold_path}/src-train.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.corpus_1, 'path_tgt'): onmtcfg.data.corpus_1.path_tgt = f'{fold_path}/tgt-train.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.valid, 'path_src'): onmtcfg.data.valid.path_src = f'{fold_path}/src-valid.txt'
            if OmegaConf.is_interpolation(onmtcfg.data.valid, 'path_tgt'): onmtcfg.data.valid.path_tgt = f'{fold_path}/tgt-valid.txt'
            if OmegaConf.is_interpolation(onmtcfg, 'src_vocab'): onmtcfg.src_vocab = f'{fold_path}/vocab.src'
            if OmegaConf.is_interpolation(onmtcfg, 'tgt_vocab'): onmtcfg.tgt_vocab = f'{fold_path}/vocab.tgt'
            if OmegaConf.is_interpolation(onmtcfg, 'save_data'): onmtcfg.save_data = f'{fold_path}/'
            if OmegaConf.is_interpolation(onmtcfg, 'save_model'): onmtcfg.save_model = f'{fold_path}/model'

            log.info(f'{opentf.textcolor["green"]}Overriding onmt.data config for fold{foldidx} in {fold_path}/config.yml ...{opentf.textcolor["reset"]}')
            OmegaConf.save(onmtcfg, f'{fold_path}/config.yml', resolve=False)

            # cli_cmd = f'onmt_build_vocab -config {fold_path}/config.yml -n_sample {len(input_data)}'
            # log.info(cli_cmd)
            # p = subprocess.Popen(shlex.split(cli_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # for l in p.stdout: log.info("[onmt_build_vocab stdout] %s", l.strip())
            # for l in p.stderr: log.error("[onmt_build_vocab stderr] %s", l.strip())
            # p.wait()

            #NOTE: it overrides the main command
            onmt_build = opentf.install_import('', 'onmt.bin.build_vocab', 'main')
            sys.argv += ['onmt_build_vocab', '-config', f'{fold_path}/config.yml']
            onmt_build()

        with open(f'{self.output}/src-test.txt', 'w') as src_test: src_test.writelines(input_data[splits['test']])
        with open(f'{self.output}/tgt-test.txt', 'w') as tgt_test: tgt_test.writelines(output_data[splits['test']])

    def learn(self, teamsvecs, splits, prev_model):
        self._prep(teamsvecs, splits)
        # return
        onmt_train = opentf.install_import('', 'onmt.bin.train', 'main')
        for foldidx in splits['folds'].keys():
            fold_path = f'{self.output}/f{foldidx}'
            train_size = len(splits['folds'][foldidx]['train'])
            onmtcfg = OmegaConf.load(f'{fold_path}/config.yml')

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

            log.info(f'{opentf.textcolor["blue"]}Overriding onmt config for train for fold{foldidx} in {fold_path}/config.yml ...{opentf.textcolor["reset"]}')
            OmegaConf.save(onmtcfg, f'{fold_path}/config.yml', resolve=True)

            # cli_cmd = f'onmt_train  -config {self.output}/f{foldidx}/config.yml '
            # log.info(cli_cmd)
            # p = subprocess.Popen(shlex.split(cli_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # for l in p.stdout: log.info("[onmt_train stdout] %s", l.strip())
            # for l in p.stderr: log.error("[onmt_train stderr] %s", l.strip())
            # p.wait()

            #NOTE: it overrides the main command
            sys.argv += ['onmt_train', '-config', f'{fold_path}/config.yml']
            onmt_train()

    #todo: per_trainstep => per_epoch
    #todo: eval on prediction files
    def test(self, teamsvecs, splits, on_train, per_epoch):
        return
        for foldidx in splits['folds'].keys():
            fold_path = f'{self.output}/fold{foldidx}'
            onmtcfg = OmegaConf.load(f'{fold_path}/config.yml')
            modelfiles = [f"{fold_path}/model_step_{onmtcfg.train_epochs}.pt"]
            if per_epoch: modelfiles += [f'{fold_path}/{_}' for _ in os.listdir(fold_path) if re.match(f'model_epoch_\d+.pt', _)]
            for model in modelfiles:
                epoch = model.split('.')[-2].split('/')[-1].split('_')[-1]
                cli_cmd = 'onmt_translate '
                cli_cmd += f'-model {model} '
                cli_cmd += f'-src {path}/src-test.txt '
                cli_cmd += f'-output {path}/fold{foldidx}/test.fold{foldidx}.epoch{epoch}.pred.csv '
                cli_cmd += f'-gpu {self.device.split(":")[1] if ":" in self.device else "0"}' if "cuda" in self.device else ''
                cli_cmd += '--min_length 2 '
                cli_cmd += '-verbose '
                print(f'{cli_cmd}')
                subprocess.Popen(shlex.split(cli_cmd)).wait()
    
    def evaluate(self, teamsvecs, splits, on_train=False, per_epoch=False, per_instance=False, metrics={}):
        return
        import evl.metric as metric
        fold_mean = pd.DataFrame()
        test_size = y_test.shape[0]
        for foldidx in splits['folds'].keys():
            fold_path = f'{path}/fold{foldidx}'
            onmtcfg = OmegaConf.load(f'{fold_path}/config.yml')
            modelfiles = [f"{fold_path}/model_step_{onmtcfg.train_epochs}.pt"]

            #per_epoch depends on "save_checkpoint_steps" param in nmt_config.yaml since it simply collect the checkpoint files
            #so, per_epoch => per_checkpoints
            if per_epoch: modelfiles += [f'{fold_path}/{_}' for _ in os.listdir(fold_path) if re.match(f'model_step_\d+.pt', _)]

            for model in modelfiles:
                epoch = model.split('.')[-2].split('/')[-1].split('_')[-1]
                pred_path = f'{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.csv'
                pred_csv = pd.read_csv(f'{pred_path}', header=None)
                # tgt_csv = pd.read_csv(f'{path}/tgt-test.txt', header=None)
                
                Y_ = np.zeros((test_size, member_count))
                # Y = np.zeros((test_size, member_count))
                for i in range(test_size):
                    yhat_list = (pred_csv.iloc[i])[0].replace('m', '').replace('<unk>', '').split()
                    yhat_count = len(yhat_list)
                    if yhat_count != 0:
                        for pred in yhat_list:
                            Y_[i, int(pred)] = 1/yhat_count

                    # y_list = (tgt_csv.iloc[i])[0].replace('m', '').split(' ')
                    # y_count = len(y_list)
                    # for tgt in y_list:
                    #     Y[i, int(tgt)] = 1
                df, df_mean, (fpr, tpr) = metric.calculate_metrics(y_test, Y_, False)
                df_mean.to_csv(f'{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.eval.mean.csv')
                with open(f'{path}/f{foldidx}.test.pred.eval.roc.pkl', 'wb') as outfile:
                    pickle.dump((fpr, tpr), outfile)
                fold_mean = pd.concat([fold_mean, df_mean], axis=1)
        fold_mean.mean(axis=1).to_frame('mean').to_csv(f'{path}/test.epoch{epoch}.pred.eval.mean.csv')


