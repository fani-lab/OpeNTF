from base64 import encode
from calendar import EPOCH
from matplotlib.pyplot import step
import yaml, pickle
import subprocess, os, re
import shlex
import numpy as np
from eval.metric import *
from mdl.ntf import Ntf
from sklearn.model_selection import KFold

from cmn.tools import NumpyArrayEncoder

class tNmt(Ntf):
    def __init__(self, tfold, step_ahead):
        super(Ntf, self).__init__()
        self.tfold = tfold
        self.step_ahead = step_ahead


    def prepare_data(self, vecs, indexes):
        input_data = []
        output_data = []
        for i, id in enumerate(vecs['id']):
            input_data.append([f's{str(skill_idx)}' for skill_idx in vecs['skill'][i].nonzero()[1]])
            # + [f"dt{str(indexes['dt2i'][indexes['i2tdt'][id[0, 0]]])}"]
            output_data.append([f'm{str(member_idx)}' for member_idx in vecs['member'][i].nonzero()[1]])
        return input_data, output_data

    def build_vocab(self, input_data, output_data, splits, settings, model_path, indexes):
        endl = "\n"
        input_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in input_data]])
        output_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in output_data]])
        epochs = settings['train_steps']
        year_idx = indexes['i2y']
        for i, v in enumerate(year_idx[:-self.step_ahead]):#the last years are for test.
            skf = KFold(n_splits=self.tfold, random_state=0, shuffle=True)
            train = np.arange(year_idx[i][0], year_idx[i + 1][0])
            for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
                splits['folds'][k]['train'] = train[trainIdx]
                splits['folds'][k]['valid'] = train[validIdx]
                year_path = f'{model_path}/{year_idx[i][1]}'
                if not os.path.isdir(year_path): os.makedirs(year_path)
                with open(f'{year_path}/splits.json', 'w') as f: json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
                fold_path = f'{year_path}/fold{k}'
                if not os.path.isdir(fold_path): os.makedirs(fold_path)
                with open(f"{fold_path}/src-train.txt", "w") as src_train: src_train.writelines(input_data[splits['folds'][k]['train']])
                with open(f"{fold_path}/src-valid.txt", "w") as src_val: src_val.writelines(input_data[splits['folds'][k]['valid']])
                with open(f"{fold_path}/tgt-train.txt", "w") as tgt_train: tgt_train.writelines(output_data[splits['folds'][k]['train']])
                with open(f"{fold_path}/tgt-valid.txt", "w") as tgt_val: tgt_val.writelines(output_data[splits['folds'][k]['valid']])

                settings['data']['corpus_1']['path_src'] = f'{fold_path}/src-train.txt'
                settings['data']['corpus_1']['path_tgt'] = f'{fold_path}/tgt-train.txt'
                settings['data']['valid']['path_src'] = f'{fold_path}/src-valid.txt'
                settings['data']['valid']['path_tgt'] = f'{fold_path}/tgt-valid.txt'
                settings['src_vocab'] = f'{fold_path}/vocab.src'
                settings['tgt_vocab'] = f'{fold_path}/vocab.tgt'
                settings['save_data'] = f'{fold_path}/'
                settings['save_model'] = f'{fold_path}/model'
                
                if i != 0:
                    settings['train_from'] = f"{model_path}/{year_idx[i-1][1]}/fold{k}/model_step_{settings['train_steps'] - epochs}.pt"
                with open(f'{fold_path}/config.yml', 'w') as outfile: yaml.safe_dump(settings, outfile)

                cli_cmd = 'onmt_build_vocab '
                cli_cmd += f'-config {fold_path}/config.yml '
                cli_cmd += f'-n_sample {len(input_data)}'
                print(f'{cli_cmd}')
                subprocess.Popen(shlex.split(cli_cmd)).wait()
            settings['train_steps'] += epochs
        with open(f"{model_path}/src-test.txt", "w") as src_test: src_test.writelines(input_data[splits['test']])
        with open(f"{model_path}/tgt-test.txt", "w") as tgt_test: tgt_test.writelines(output_data[splits['test']])

        return model_path

    def learn(self, splits, path):
        for foldidx in splits['folds'].keys():
            cli_cmd = 'onmt_train '
            cli_cmd += f'-config {path}/fold{foldidx}/config.yml '
            print(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

    #todo: per_trainstep => per_epoch
    #todo: eval on prediction files
    def test(self, splits, path, last_train_year, per_epoch):
        for foldidx in splits['folds'].keys():
            fold_path = f'{path}/{last_train_year}/fold{foldidx}'
            with open(f'{fold_path}/config.yml') as infile: cfg = yaml.safe_load(infile)
            modelfiles = [f"{fold_path}/model_step_{cfg['train_steps']}.pt"]
            if per_epoch: modelfiles += [f'{fold_path}/{_}' for _ in os.listdir(fold_path) if re.match(f'model_step_\d+.pt', _)]
            for model in modelfiles:
                epoch = model.split('.')[-2].split('/')[-1].split('_')[-1]
                cli_cmd = 'onmt_translate '
                cli_cmd += f'-model {model} '
                cli_cmd += f'-src {path}/src-test.txt '
                cli_cmd += f'-output {path}/{last_train_year}/fold{foldidx}/test.fold{foldidx}.epoch{epoch}.pred.csv '
                cli_cmd += '-gpu 0 '
                cli_cmd += '--min_length 1 '
                cli_cmd += '-verbose '
                print(f'{cli_cmd}')
                subprocess.Popen(shlex.split(cli_cmd)).wait()
    
    def eval(self, splits, path, last_train_year, member_count, y_test, per_epoch):
        fold_mean = pd.DataFrame()
        test_size = y_test.shape[0]
        for foldidx in splits['folds'].keys():
            fold_path = f'{path}/{last_train_year}/fold{foldidx}'
            with open(f'{fold_path}/config.yml') as infile: cfg = yaml.safe_load(infile)
            modelfiles = [f"{fold_path}/model_step_{cfg['train_steps']}.pt"]
            if per_epoch: modelfiles += [f'{fold_path}/{_}' for _ in os.listdir(fold_path) if re.match(f'model_step_\d+.pt', _)]
            for model in modelfiles:
                epoch = model.split('.')[-2].split('/')[-1].split('_')[-1]
                pred_path = f'{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.csv'
                pred_csv = pd.read_csv(f'{pred_path}', header=None)
                # tgt_csv = pd.read_csv(f'{path}/tgt-test.txt', header=None)
                
                Y_ = np.zeros((test_size, member_count))
                # Y = np.zeros((test_size, member_count))
                for i in range(test_size):
                    yhat_list = (pred_csv.iloc[i])[0].replace('m', '').split(' ')
                    yhat_count = len(yhat_list)
                    if yhat_count != 0:
                        for pred in yhat_list:
                            Y_[i, int(pred)] = 1/yhat_count

                    # y_list = (tgt_csv.iloc[i])[0].replace('m', '').split(' ')
                    # y_count = len(y_list)
                    # for tgt in y_list:
                    #     Y[i, int(tgt)] = 1
                df, df_mean, (fpr, tpr) = calculate_metrics(y_test, Y_, False)
                df_mean.to_csv(f'{fold_path}/test.fold{foldidx}.epoch{epoch}.pred.eval.mean.csv')
                with open(f'{path}/f{foldidx}.test.pred.eval.roc.pkl', 'wb') as outfile:
                    pickle.dump((fpr, tpr), outfile)
                fold_mean = pd.concat([fold_mean, df_mean], axis=1)
        fold_mean.mean(axis=1).to_frame('mean').to_csv(f'{path}/test.epoch{epoch}.pred.eval.mean.csv')
                
                      
    def run(self, splits, vecs, indexes, output, settings, cmd):
        with open(settings['base_config']) as infile: base_config = yaml.safe_load(infile)

        encoder_type = base_config['encoder_type']
        learning_rate = base_config['learning_rate']
        word_vec_size = base_config['word_vec_size']
        batch_size = base_config['batch_size']
        epochs = base_config['train_steps']
        team_count = vecs['skill'].shape[0]
        skill_count = vecs['skill'].shape[1]
        member_count = vecs['member'].shape[1]

        if encoder_type == 'rnn':
            encoder_type = base_config['rnn_type']
            layer_size = base_config['rnn_size']
        elif encoder_type == 'transformer':
            layer_size = base_config['transformer_ff']

        base_config['train_from'] = None
        
        model_path = f"{output}/t{team_count}.s{skill_count}.m{member_count}.et{encoder_type}.l{layer_size}.wv{word_vec_size}.lr{learning_rate}.b{batch_size}.e{epochs}"
        if not os.path.isdir(output): os.makedirs(output)
        
        y_test = vecs['member'][splits['test']]
        year_idx = indexes['i2y']
        if 'train' in cmd:
            input_data, output_data = self.prepare_data(vecs, indexes)
            model_path = self.build_vocab(input_data, output_data, splits, base_config, model_path, indexes)
            for i, v in enumerate(year_idx[:-self.step_ahead]):
                self.learn(splits, f'{model_path}/{year_idx[i][1]}')
        if 'test' in cmd: self.test(splits, model_path, year_idx[-self.step_ahead - 1][1], per_epoch=False)
        if 'eval' in cmd: self.eval(splits, model_path, year_idx[-self.step_ahead - 1][1], member_count, y_test, per_epoch=False)
        if 'plot' in cmd: self.plot_roc(model_path, splits, False)

