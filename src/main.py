import sys,os, json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from json import JSONEncoder
from itertools import product

import param
from cmn.publication import Publication
from cmn.movie import Movie
from cmn.patent import Patent
from mdl.fnn import Fnn
from mdl.bnn import Bnn
from mdl.rnd import Rnd
from mdl.nmt import Nmt
from mdl.team2vec import Team2Vec

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def create_evaluation_splits(n_sample, n_folds, train_ratio=0.85, output='./'):
    train, test = train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=0, shuffle=True)
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        splits['folds'][k]['valid'] = train[validIdx]

    with open(f'{output}/splits.json', 'w') as f:
        json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)

    return splits

def aggregate(output):
    files = list()
    for dirpath, dirnames, filenames in os.walk(output):
        if not dirnames: files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred.eval.mean.csv")]
    files = pd.DataFrame(files, columns=['', '', 'domain', 'baseline', 'setting', 'rfile'])
    rfiles = files.groupby('rfile')
    for rf, r in rfiles:
        dfff = pd.DataFrame()
        rdomains = r.groupby('domain')
        for rd, rr in rdomains:
            names = ['metrics']
            dff = pd.DataFrame()
            df = rdomains.get_group(rd)
            hr = False
            for i, row in df.iterrows():
                if not hr:
                    dff = pd.concat([dff, pd.read_csv(f"{output}{rd}/{row['baseline']}/{row['setting']}/{rf}", usecols=[0])], axis=1, ignore_index=True)
                    hr = True
                dff = pd.concat([dff, pd.read_csv(f"{output}{rd}/{row['baseline']}/{row['setting']}/{rf}", usecols=[1])], axis=1, ignore_index=True)
                names += [row['baseline'] + '.' + row['setting']]
            dfff = pd.concat([dfff, pd.DataFrame({0: [rd]}), dff], ignore_index=True)
        dfff.set_axis(names, axis=1, inplace=True)
        dfff.to_csv(f"{output}{rf.replace('.csv', '.agg.csv')}", index=False)

def run(data_list, domain_list, filter, model_list, output, settings):
    filter_str = f".filtered.mt{settings['data']['filter']['min_nteam']}.ts{settings['data']['filter']['min_team_size']}" if filter else ""

    datasets = {}
    models = {}
    if 'dblp' in domain_list: datasets['dblp'] = Publication
    if 'imdb' in domain_list: datasets['imdb'] = Movie
    if 'uspt' in domain_list: datasets['uspt'] = Patent

    if 'random' in model_list: models['random'] = Rnd()
    if 'fnn' in model_list: models['fnn'] = Fnn()
    if 'bnn' in model_list: models['bnn'] = Bnn()
    if 'fnn_emb' in model_list: models['fnn_emb'] = Fnn()
    if 'bnn_emb' in model_list: models['bnn_emb'] = Bnn()
    if 'nmt' in model_list: models['nmt'] = Nmt()

    assert len(datasets) > 0
    assert len(datasets) == len(domain_list)
    assert len(models) > 0

    for (d_name, d_cls), (m_name, m_obj) in product(datasets.items(), models.items()):
        datapath = data_list[domain_list.index(d_name)]
        prep_output = f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}'
        vecs, indexes = d_cls.generate_sparse_vectors(datapath, f'{prep_output}{filter_str}', filter, settings['data'])
        splits = create_evaluation_splits(vecs['id'].shape[0], settings['model']['nfolds'], settings['model']['train_test_split'], output=f'{prep_output}{filter_str}')
        if m_name.find('_emb') > 0:
            t2v = Team2Vec(vecs, 'skill', f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}{filter_str}')
            emb_setting = settings['model']['baseline']['emb']
            t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
            vecs['skill'] = t2v.dv()

        m_obj.run(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}', settings['model']['baseline'][m_name.replace('_emb', '')], settings['model']['cmd'])
    aggregate(output)

def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.json)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', default=[], required=True, help='a list of domains; required; (eg. -domain dblp imdb uspt)')
    dataset.add_argument('-filter', type=int, default=0, choices=[0, 1], help='remove outliers? (e.g., -filter 0 (default) or 1)')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', '--model-list', nargs='+', default=[], required=True, help='a list of neural models (eg. -model random fnn bnn fnn_emb bnn_emb nmt)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: -output ./../output/)')


# python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json
# 						  ../data/raw/imdb/toy.title.basics.tsv
# 						  ../data/raw/uspt/toy.patent.tsv
# 					-domain dblp imdb uspt
# 					-model random fnn fnn_emb bnn bnn_emb
#                   -filter 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Team Formation')
    addargs(parser)
    args = parser.parse_args()

    run(data_list=args.data_list,
        domain_list=args.domain_list,
        filter=args.filter,
        model_list=args.model_list,
        output=args.output,
        settings=param.settings)

    # aggregate(args.output)

