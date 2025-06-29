import pickle, subprocess, os, re, shlex, numpy as np, logging
from omegaconf import OmegaConf
log = logging.getLogger(__name__)

from mdl.ntf import Ntf

class Nmt(Ntf):
    def __init__(self, output, pytorch, device, seed, cgf):
        super().__init__(output, pytorch, device, seed, cgf)
        self.openmtcfg = OmegaConf.load('./mdl/__config__.nmt.yaml')

    def _prep(self, teamsvecs, splits):
        input_data = []
        output_data = []
        for i in range(teamsvecs['skill'].shape[0]): #n_teams
            input_data.append([f's{str(skill_idx)}' for skill_idx in teamsvecs['skill'][i].nonzero()[1]])
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

            self.openmtcfg.data.corpus_1.path_src = f'{fold_path}/src-train.txt'
            self.openmtcfg.data.corpus_1.path_tgt = f'{fold_path}/tgt-train.txt'
            self.openmtcfg.data.valid.path_src = f'{fold_path}/src-valid.txt'
            self.openmtcfg.data.valid.path_tgt = f'{fold_path}/tgt-valid.txt'
            self.openmtcfg.src_vocab = f'{fold_path}/vocab.src'
            self.openmtcfg.tgt_vocab = f'{fold_path}/vocab.tgt'
            self.openmtcfg.save_data = f'{fold_path}/'
            self.openmtcfg.save_model = f'{fold_path}/model'

            self.openmtcfg.world_size = 1
            self.openmtcfg.gpu_ranks = ([self.device.split(':')[1]] if 'cuda' in self.device else []) if '_{acceleration}' == self.openmtcfg.gpu_ranks else self.openmtcfg.gpu_ranks
            self.openmtcfg.seed = self.seed
            self.openmtcfg.train_epochs = self.cfg.e
            # self.openmtcfg.save_checkpoint_steps = self.cfg.spe
            self.openmtcfg.batch_size = self.cfg.b
            self.openmtcfg.learning_rate = self.cfg.lr
            self.openmtcfg.early_stopping = self.cfg.es
            self.openmtcfg.encoder_type = self.cfg.enc

            OmegaConf.save(self.openmtcfg, f'{fold_path}/config.yml', resolve=True)

            cli_cmd = 'onmt_build_vocab '
            cli_cmd += f'-config {fold_path}/config.yml '
            cli_cmd += f'-n_sample {len(input_data)}'
            log.info(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

        with open(f"{self.output}/src-test.txt", "w") as src_test: src_test.writelines(input_data[splits['test']])
        with open(f"{self.output}/tgt-test.txt", "w") as tgt_test: tgt_test.writelines(output_data[splits['test']])

    def learn(self, teamsvecs, splits, prev_model):
        self._prep(teamsvecs, splits)
        for foldidx in splits['folds'].keys():
            cli_cmd = 'onmt_train '
            cli_cmd += f'-config {self.output}/f{foldidx}/config.yml '
            log.info(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

    #todo: per_trainstep => per_epoch
    #todo: eval on prediction files
    def test(self, teamsvecs, splits, on_train, per_epoch):
        for foldidx in splits['folds'].keys():
            fold_path = f'{path}/fold{foldidx}'
            self.openmtcfg = OmegaConf.load(f'{fold_path}/config.yml')
            modelfiles = [f"{fold_path}/model_step_{self.openmtcfg.train_epochs}.pt"]
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
    
    def eval(self, splits, path, member_count, y_test, per_epoch):
        import evl.metric as metric
        fold_mean = pd.DataFrame()
        test_size = y_test.shape[0]
        for foldidx in splits['folds'].keys():
            fold_path = f'{path}/fold{foldidx}'
            self.openmtcfg = OmegaConf.load(f'{fold_path}/config.yml')
            modelfiles = [f"{fold_path}/model_step_{self.openmtcfg.train_epochs}.pt"]

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
                      
    def run(self, splits, vecs, indexes, output, settings, cmd):
        with open(settings['base_config']) as infile: base_config = yaml.safe_load(infile)

        learning_rate = base_config['learning_rate']
        word_vec_size = base_config['word_vec_size']
        batch_size = base_config['batch_size']
        epochs = base_config['train_steps']


        
        y_test = vecs['member'][splits['test']]
        
        if 'train' in cmd:
            model_path = self.build_vocab(splits, base_config, model_path)
            self.learn(splits, model_path)
        if 'test' in cmd: self.test(splits, model_path, per_epoch=True)
        if 'eval' in cmd: self.eval(splits, model_path, member_count, y_test, per_epoch=True)
        if 'plot' in cmd: self.plot_roc(model_path, splits, False)

