import sys,os, json
import argparse
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from json import JSONEncoder

import param
from cmn.publication import Publication
from cmn.movie import Movie
from cmn.patent import Patent
import fnn_main
import bnn_main
import nmt_main
from mdl.team2vec import Team2Vec

sys.path.extend(['../cmn'])

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

def run(datapath, domain, filter, model, output, settings):
    prep_output = f'./../data/preprocessed/{domain}/{os.path.split(datapath)[-1]}'
    if domain == 'dblp':
        vecs, indexes = Publication.generate_sparse_vectors(datapath, prep_output, filter, settings['data'])

    if domain == 'imdb':
        vecs, indexes = Movie.generate_sparse_vectors(datapath, prep_output, filter, settings['data'])

    if domain == 'uspt':
        vecs, indexes = Patent.generate_sparse_vectors(datapath, prep_output, filter, settings['data'])

    splits = create_evaluation_splits(len(indexes['t2i']), settings['model']['nfolds'], settings['model']['train_test_split'], output=prep_output)

    filter_str = f".filtered.mt{settings['data']['filter']['min_nteam']}.ts{settings['data']['filter']['min_team_size']}" if filter else None
    if model == 'fnn':
        fnn_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}{filter_str}/fnn', settings['model']['baseline']['fnn'], settings['model']['cmd'])

    if model == 'bnn':
        bnn_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}{filter_str}/bnn', settings['model']['baseline']['fnn'], settings['model']['cmd'])

    if model == 'fnn_emb':
        t2v = Team2Vec(vecs, 'skill', f'./../data/preprocessed/{domain}/{os.path.split(datapath)[-1]}{filter_str}')
        emb_setting = settings['model']['baseline']['emb']
        t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
        vecs['skill'] = t2v.dv()
        fnn_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}/fnn_emb', settings['model']['baseline']['fnn'], settings['model']['cmd'])

    if model == 'bnn_emb':
        t2v = Team2Vec(vecs, 'skill', f'./../data/preprocessed/{domain}/{os.path.split(datapath)[-1]}{filter_str}')
        emb_setting = settings['model']['baseline']['emb']
        t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
        vecs['skill'] = t2v.dv()
        bnn_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}/bnn_emb', settings['model']['baseline']['fnn'], settings['model']['cmd'])

    if model == 'nmt':
        nmt_main.main(splits, vecs, indexes, f'{output}{os.path.split(datapath)[-1]}/nmt', './nmt_base_config.yaml', cmd=['train', 'test', 'eval'])

def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', type=str, required=True, help='The dataset path; required; (example: ./../data/raw/toy.json)')
    dataset.add_argument('-domain', type=str, required=True, choices=['dblp', 'imdb', 'uspt'], help='The dataset path; required; (example: dblp)')
    dataset.add_argument('-filter', type=int, default=0, choices=[0, 1], help='Remove outliers? (0=False(default), 1=True)')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', type=str, required=True, choices=['fnn', 'bnn', 'fnn_emb', 'bnn_emb', 'nmt'], help='The model name (example: fnn)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: ./../output/)')

# python -u main.py -data=./../data/raw/dblp/toy.dblp.v12.json -domain=dblp -model=fnn  2>&1 | tee ./../output/toy.log &
# python -u main.py -data=./../data/raw/dblp/dblp.v12.json -domain=dblp -model=fnn 2>&1 | tee ./../output/dblp.log &
# python -u main.py -data=./../data/raw/imdb/toy.title.basics.tsv -domain=imdb -model=fnn
# python -u main.py -data=./../data/raw/imdb/title.basics.tsv -domain=imdb -model=fnn
# python -u main.py -data=./../data/raw/uspt/toy.patent.tsv -domain=uspt -model=fnn

global ncores
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Team Formation')
    addargs(parser)
    args = parser.parse_args()

    run(datapath=args.data,
        domain=args.domain,
        filter=args.filter,
        model=args.model,
        output=args.output,
        settings=param.settings)

