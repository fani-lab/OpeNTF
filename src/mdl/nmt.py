import yaml
import subprocess, os
import shlex
import numpy as np

from mdl.ntf import Ntf

class Nmt(Ntf):
    def __init__(self):
        super(Ntf, self).__init__()

    def prepare_data(self, vecs):
        src_skill = []
        tgt_member = []
        for i, id in enumerate(vecs['id']):
            src_skill.append([f's{str(skill_idx)}' for skill_idx in vecs['skill'][i].nonzero()[1]])
            tgt_member.append([f'm{str(member_idx)}' for member_idx in vecs['member'][i].nonzero()[1]])
        return src_skill, tgt_member

    def build_vocab(self, input_data, output_data, splits, settings, model_path):
        endl = "\n"
        input_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in input_data]])
        output_data = np.array(['{}{}'.format(_, endl) for _ in [' '.join(_) for _ in output_data]])

        for foldidx in splits['folds'].keys():
            fold_path = f'{model_path}/fold{foldidx}'
            if not os.path.isdir(fold_path): os.makedirs(fold_path)

            with open(f"{fold_path}/src-train.txt", "w") as src_train: src_train.writelines(input_data[splits['folds'][foldidx]['train']])
            with open(f"{fold_path}/src-valid.txt", "w") as src_val: src_val.writelines(input_data[splits['folds'][foldidx]['valid']])
            with open(f"{fold_path}/tgt-train.txt", "w") as tgt_train: tgt_train.writelines(output_data[splits['folds'][foldidx]['train']])
            with open(f"{fold_path}/tgt-valid.txt", "w") as tgt_val: tgt_val.writelines(output_data[splits['folds'][foldidx]['valid']])

            settings['data']['corpus_1']['path_src'] = f'{fold_path}/src-train.txt'
            settings['data']['corpus_1']['path_tgt'] = f'{fold_path}/tgt-train.txt'
            settings['data']['valid']['path_src'] = f'{fold_path}/src-valid.txt'
            settings['data']['valid']['path_tgt'] = f'{fold_path}/tgt-valid.txt'
            settings['src_vocab'] = f'{fold_path}/vocab.src'
            settings['tgt_vocab'] = f'{fold_path}/vocab.tgt'
            settings['save_data'] = f'{fold_path}/'
            settings['save_model'] = f'{fold_path}/model'

            with open(f'{fold_path}/config.yml', 'w') as outfile: yaml.safe_dump(settings, outfile)

            cli_cmd = 'onmt_build_vocab '
            cli_cmd += f'-config {fold_path}/config.yml '
            cli_cmd += f'-n_sample {len(input_data)}'
            print(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

        with open(f"{model_path}/src-test.txt", "w") as src_test: src_test.writelines(input_data[splits['test']])
        with open(f"{model_path}/tgt-test.txt", "w") as tgt_test: tgt_test.writelines(output_data[splits['test']])

        return model_path

    def learn(self, splits, path):
        for foldidx in splits['folds'].keys():
            cli_cmd = 'onmt_train '
            cli_cmd += f'-config {path}/fold{foldidx}/config.yml '
            print(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

    def test(self, splits, path):
        for foldidx in splits['folds'].keys():
            cli_cmd = 'onmt_translate '
            cli_cmd += f'-model {path}/fold{foldidx}/model_step_100.pt '
            cli_cmd += f'-src {path}/src-test.txt '
            cli_cmd += f'-output {path}/test{foldidx}.pred.csv '
            cli_cmd += '-gpu 0 '
            cli_cmd += '-verbose '
            print(f'{cli_cmd}')
            subprocess.Popen(shlex.split(cli_cmd)).wait()

    def run(self, splits, vecs, indexes, output, settings, cmd):
        with open(settings['base_config']) as infile:
            base_config = yaml.safe_load(infile)

        model_path = f"{output}/t{vecs['skill'].shape[0]}.s{vecs['skill'].shape[1]}.m{vecs['member'].shape[1]}"
        if not os.path.isdir(output): os.makedirs(output)

        if 'train' in cmd:
            input_data, output_data = self.prepare_data(vecs)
            model_path = self.build_vocab(input_data, output_data, splits, base_config, model_path)
            self.learn(splits, model_path)
        if 'test' in cmd : self.test(splits, model_path)