import sys,os, json
import argparse
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from json import JSONEncoder
from itertools import product

import param
from cmn.publication import Publication
from cmn.movie import Movie
from cmn.patent import Patent
from mdl.ntf import Ntf
from mdl.tntf import TNtf
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

def create_streaming_splits(n_sample, train_ratio, output='./'):
    train, test = train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=0, shuffle=False)
    splits = dict()
    splits['train'] = train
    splits['test'] = test
    with open(f'{output}/temporal_splits.json', 'w') as f:
        json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)

    return splits

def run(data_list, domain_list, filter, model_list, output, settings):
    filter_str = f".filtered.mt{settings['data']['filter']['min_nteam']}.ts{settings['data']['filter']['min_team_size']}" if filter else ""

    datasets = {}
    models = {}
    bayes = settings['model']['baseline']['fnn']['bayesian']

    if 'dblp' in domain_list: datasets['dblp'] = Publication
    if 'imdb' in domain_list: datasets['imdb'] = Movie
    if 'uspt' in domain_list: datasets['uspt'] = Patent

    if 'random' in model_list: models['random'] = Rnd()

    if not bayes:
        if not settings['data']['temporal']:
            if 'fnn' in model_list: models['fnn'] = Ntf()
            if 'fnn_emb' in model_list: models['fnn_emb'] = Ntf()
        else:
            if 'tfnn' in model_list: models['tfnn'] = TNtf()
            if 'tfnn_emb' in model_list: models['tfnn_emb'] = TNtf()
    else:
        if not settings['data']['temporal']:
            if 'bnn' in model_list: models['bnn'] = Ntf()
            if 'bnn_emb' in model_list: models['bnn_emb'] = Ntf()
        else:
            if 'tbnn' in model_list and bayes: models['tbnn'] = TNtf()
            if 'tbnn_emb' in model_list: models['tbnn_emb'] = TNtf()
  
    if 'nmt' in model_list: models['nmt'] = Nmt()

    for (d_name, d_cls), (m_name, m_obj) in product(datasets.items(), models.items()):
        datapath = data_list[domain_list.index(d_name)]
        prep_output = f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}'
        vecs, indexes = d_cls.generate_sparse_vectors(datapath, f'{prep_output}{filter_str}', filter, settings['data'])

        if settings['data']['temporal']:
            splits = create_streaming_splits(vecs['id'].shape[0], settings['model']['train_test_split'], output=f'{prep_output}{filter_str}')
        else:
            splits = create_evaluation_splits(vecs['id'].shape[0], settings['model']['nfolds'], settings['model']['train_test_split'], output=f'{prep_output}{filter_str}')
        
        if m_name.find('_emb') > 0:
            t2v = Team2Vec(vecs, 'skill', f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}{filter_str}', settings['data']['temporal'])
            emb_setting = settings['model']['baseline']['emb']
            t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
            vecs['skill'] = t2v.dv()

        m_obj.run(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}', settings['model']['baseline'][m_name.replace('t', '').replace('_emb', '').replace('bnn', 'fnn')], settings['model']['cmd'])

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

