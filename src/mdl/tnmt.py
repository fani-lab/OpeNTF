import yaml, pickle, json
import subprocess, os, re
import shlex
import numpy as np
from sklearn.model_selection import KFold
import torch

from eval.metric import *
from mdl.nmt import Nmt
from cmn.tools import NumpyArrayEncoder

class tNmt(Nmt):
    def __init__(self, tfold, step_ahead):
        super(Nmt, self).__init__()
        self.tfold = tfold
        self.step_ahead = step_ahead

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

                settings['world_size'] = 1
                settings['gpu_ranks'] = [0] if torch.cuda.is_available() else []

                settings['save_checkpoint_steps'] = 500  # overrides per_epoch evaluation and it becomes per_checkpoints

                if i != 0:
                    settings['train_from'] = f"{model_path}/{year_idx[i-1][1]}/fold{k}/model_step_{settings['train_steps'] - epochs}.pt"
                with open(f'{fold_path}/config.yml', 'w') as outfile: yaml.safe_dump(settings, outfile)

                cli_cmd = 'onmt_build_vocab '
                cli_cmd += f'-config {fold_path}/config.yml '
                cli_cmd += f'-n_sample {len(input_data)}'
                print(f'{cli_cmd}')
                subprocess.Popen(shlex.split(cli_cmd)).wait()
            settings['train_steps'] += epochs
        with open(f"{model_path}/{year_idx[-self.step_ahead - 1][1]}/src-test.txt", "w") as src_test: src_test.writelines(input_data[splits['test']])
        with open(f"{model_path}/{year_idx[-self.step_ahead - 1][1]}/tgt-test.txt", "w") as tgt_test: tgt_test.writelines(output_data[splits['test']])

        return model_path

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
            input_data, output_data = self.prepare_data(vecs)
            model_path = self.build_vocab(input_data, output_data, splits, base_config, model_path, indexes)
            for i, v in enumerate(year_idx[:-self.step_ahead]):
                self.learn(splits, f'{model_path}/{year_idx[i][1]}')
        if 'test' in cmd: self.test(splits, f'{model_path}/{year_idx[-self.step_ahead - 1][1]}', per_epoch=False)
        if 'eval' in cmd: self.eval(splits, f'{model_path}/{year_idx[-self.step_ahead - 1][1]}', member_count, y_test, per_epoch=False)
        if 'plot' in cmd: self.plot_roc(f'{model_path}/{year_idx[-self.step_ahead - 1][1]}', splits, False)

