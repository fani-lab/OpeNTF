import yaml
import subprocess, os
import shlex
import numpy as np

def prepare_data(teams):
    input_data = []
    output_data = []
    for team in teams:
        input_data.append(" ".join(team.get_skills()))
        output_data.append(" ".join(team.get_members_names()))
    return input_data, output_data

def build_vocab(input_data, output_data, splits):
    endl = "\n"
    input_data = np.array(["{}{}".format(i,endl) for i in input_data])
    output_data = np.array(["{}{}".format(i,endl) for i in output_data])
    size = len(input_data)

    config_path = 'base_config.yaml'
    with open(config_path) as infile:
        base_config = yaml.safe_load(infile)

    enc_type = base_config['encoder_type']
    dec_type = base_config['decoder_type']
    rnn_size = base_config['rnn_size']
    layers = base_config['layers']
    batch_size = base_config['batch_size']

    nmt_path = '../output/nmt'
    if not os.path.isdir(nmt_path): os.mkdir(nmt_path)

    model_path = f'{nmt_path}/{enc_type}_{dec_type}_{rnn_size}_{layers}_{batch_size}_{size}'
    if not os.path.isdir(model_path): os.mkdir(model_path)

    for foldidx in splits['folds'].keys():
        fold_path = f'{model_path}/fold{foldidx}'
        if not os.path.isdir(fold_path): os.mkdir(fold_path)

        with open(f"{fold_path}/src-train.txt", "w") as src_train:
            src_train.writelines(input_data[splits['folds'][foldidx]['train']])
        with open(f"{fold_path}/src-valid.txt", "w") as src_val:
            src_val.writelines(input_data[splits['folds'][foldidx]['valid']])           
        with open(f"{fold_path}/tgt-train.txt", "w") as tgt_train:
            tgt_train.writelines(output_data[splits['folds'][foldidx]['train']])
        with open(f"{fold_path}/tgt-valid.txt", "w") as tgt_val:
            tgt_val.writelines(output_data[splits['folds'][foldidx]['valid']])
          
    with open(f"{model_path}/src-test.txt", "w") as src_test:  
            src_test.writelines(input_data[splits['test']])
    with open(f"{model_path}/tgt-test.txt", "w") as tgt_test:  
            tgt_test.writelines(output_data[splits['test']])         

    for foldidx in splits['folds'].keys():
        fold_path = f'{model_path}/fold{foldidx}'
        if not os.path.isdir(fold_path): os.mkdir(fold_path)

        base_config['data']['corpus_1']['path_src'] = f'{fold_path}/src-train.txt'
        base_config['data']['corpus_1']['path_tgt'] = f'{fold_path}/tgt-train.txt'
        base_config['data']['valid']['path_src'] = f'{fold_path}/src-valid.txt'
        base_config['data']['valid']['path_tgt'] = f'{fold_path}/tgt-valid.txt'
        base_config['src_vocab'] = f'{fold_path}/vocab.src'
        base_config['tgt_vocab'] = f'{fold_path}/vocab.tgt'
        base_config['save_data'] = f'{fold_path}/'
        base_config['save_model'] = f'{fold_path}/model'

        with open(f'{fold_path}/config.yml', 'w') as outfile:
            yaml.safe_dump(base_config, outfile)

        cli_cmd = 'onmt_build_vocab '
        cli_cmd += f'-config {fold_path}/config.yml '
        cli_cmd += f'-n_sample {size}'
        print(f'{cli_cmd}')
        subprocess.Popen(shlex.split(cli_cmd)).wait()
    return model_path


def learn(splits, path):
    for foldidx in splits['folds'].keys():
        cli_cmd = 'onmt_train '
        cli_cmd += f'-config {path}/fold{foldidx}/config.yml '
        print(f'{cli_cmd}')
        subprocess.Popen(shlex.split(cli_cmd)).wait()

def test(splits, path):
    for foldidx in splits['folds'].keys():
        cli_cmd = 'onmt_translate '
        cli_cmd += f'-model {path}/fold{foldidx}/model_step_100.pt '
        cli_cmd += f'-src {path}/src-test.txt '
        cli_cmd += f'-output {path}/test{foldidx}.pred.csv '
        cli_cmd += '-gpu 0 '
        cli_cmd += '-verbose '
        print(f'{cli_cmd}')
        subprocess.Popen(shlex.split(cli_cmd)).wait()

def main(splits, input_data, output_data, cmd=['train', 'eval', 'test']):
    if 'train' in cmd:
        # input_data, output_data = prepare_data(teams)
        model_path = build_vocab(input_data, output_data, splits)
        learn(splits, model_path)
    if 'test' in cmd :test(splits, model_path)