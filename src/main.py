import sys,os, json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from itertools import product
from scipy.sparse import lil_matrix
import scipy.sparse
import param
from cmn.tools import NumpyArrayEncoder
from cmn.publication import Publication
from cmn.movie import Movie
from cmn.patent import Patent
from mdl.fnn import Fnn
from mdl.bnn import Bnn
from mdl.rnd import Rnd
from mdl.nmt import Nmt
from mdl.tnmt import tNmt
from mdl.tntf import tNtf
from mdl.team2vec import Team2Vec

def create_evaluation_splits(n_sample, n_folds, train_ratio=0.85, year_idx=None, output='./', step_ahead=1):   
    if year_idx:
        train = np.arange(year_idx[0][0], year_idx[-step_ahead][0])  # for teamporal folding, we do on each time interval ==> look at tntf.py
        test = np.arange(year_idx[-step_ahead][0], n_sample)
    else:
        train, test = train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=0, shuffle=True)
        
    splits = dict()
    splits['test'] = test
    splits['folds'] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits['folds'][k] = dict()
        splits['folds'][k]['train'] = train[trainIdx]
        splits['folds'][k]['valid'] = train[validIdx]

    with open(f'{output}/splits.json', 'w') as f: json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
    return splits

def aggregate(output):
    files = list()
    for dirpath, dirnames, filenames in os.walk(output):
        if not dirnames: files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred.eval.mean.csv")]

    #concate the year folder to setting for temporal baselines
    for file in files:
        if file[3].startswith('t'):
            file[4] += '/' + file[5]
            del file[5]

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
            dff.set_axis(names, axis=1, inplace=True)
            dff.to_csv(f"{output}{rd}/{rf.replace('.csv', '.agg.csv')}", index=False)

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

    #temporal baselines
    if 'tfnn' in model_list: models['tfnn'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn' in model_list: models['tbnn'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_emb' in model_list: models['tfnn_emb'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_emb' in model_list: models['tbnn_emb'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_a1' in model_list: models['tfnn_a1'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_a1' in model_list: models['tbnn_a1'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_emb_a1' in model_list: models['tfnn_emb_a1'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_emb_a1' in model_list: models['tbnn_emb_a1'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_dt2v_emb' in model_list: models['tfnn_dt2v_emb'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_dt2v_emb' in model_list: models['tbnn_dt2v_emb'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tnmt' in model_list: models['tnmt'] = tNmt(settings['model']['nfolds'], settings['model']['step_ahead'])

    assert len(datasets) > 0
    assert len(datasets) == len(domain_list)
    assert len(models) > 0

    temporal = any(m_name.startswith('t') for (m_name, m_obj) in models.items())
    for (d_name, d_cls) in datasets.items():
        datapath = data_list[domain_list.index(d_name)]
        prep_output = f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}'
        vecs, indexes = d_cls.generate_sparse_vectors(datapath, f'{prep_output}{filter_str}', filter, settings['data'])
        year_idx = []
        for i in range(1, len(indexes['i2y'])):
            if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > settings['model']['nfolds']:
                year_idx.append(indexes['i2y'][i-1])
        year_idx.append(indexes['i2y'][-1])
        indexes['i2y'] = year_idx
        splits = create_evaluation_splits(vecs['id'].shape[0], settings['model']['nfolds'], settings['model']['train_test_split'], indexes['i2y'] if temporal else None, output=f'{prep_output}{filter_str}', step_ahead=settings['model']['step_ahead'])

        for (m_name, m_obj) in models.items():
            vecs_ = vecs.copy()
            if m_name.find('_emb') > 0:
                t2v = Team2Vec(vecs, indexes, 'dt2v' if m_name.find('_dt2v') > 0 else 'skill', f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}{filter_str}')
                emb_setting = settings['model']['baseline']['emb']
                t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
                vecs_['skill'] = t2v.dv()

            if m_name.endswith('a1'): vecs_['skill'] = lil_matrix(scipy.sparse.hstack((vecs_['skill'], lil_matrix(np.ones((vecs_['skill'].shape[0], 1))))))

            baseline_name = m_name.lstrip('t').replace('_emb', '').replace('_dt2v', '').replace('_a1', '')
            print(f'Running for (dataset, model): ({d_name}, {m_name}) ... ')
            m_obj.run(splits, vecs_, indexes, f'{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}', settings['model']['baseline'][baseline_name], settings['model']['cmd'])
    if 'agg' in settings['model']['cmd']: aggregate(output)

def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.json)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str.lower, default=[], required=True, help='a list of domains; required; (eg. -domain dblp imdb uspt)')
    dataset.add_argument('-filter', type=int, default=0, choices=[0, 1], help='remove outliers? (e.g., -filter 0 (default) or 1)')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', '--model-list', nargs='+', type=str.lower, default=[], required=True, help='a list of neural models (eg. -model random fnn bnn fnn_emb bnn_emb nmt)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: -output ./../output/)')


# python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json
# 						  ../data/raw/imdb/toy.title.basics.tsv
# 						  ../data/raw/uspt/toy.patent.tsv
# 					-domain dblp imdb uspt
# 					-model random
# 					       fnn fnn_emb bnn bnn_emb nmt
# 					       tfnn tbnn tnmt tfnn_emb tbnn_emb tfnn_a1 tbnn_a1 tfnn_emb_a1 tbnn_emb_a1 tfnn_dt2v_emb tbnn_dt2v_emb
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

