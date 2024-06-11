import os, json
import argparse
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from scipy.sparse import lil_matrix
from shutil import copyfile
import scipy.sparse

import param
from cmn.tools import NumpyArrayEncoder, popular_nonpopular_ratio
from cmn.publication import Publication
from cmn.movie import Movie
from cmn.patent import Patent
from cmn.github import Repo
from mdl.fnn import Fnn
from mdl.bnn import Bnn
from mdl.rnd import Rnd
from mdl.nmt import Nmt
from mdl.tnmt import tNmt
from mdl.tntf import tNtf
from mdl.team2vec.team2vec import Team2Vec
from mdl.caser import Caser
from mdl.rrn import Rrn
from cmn.tools import generate_popular_and_nonpopular

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

def run(data_list, domain_list, fair, filter, future, augment, model_list, output, exp_id, settings):
    filter_str = f".filtered.mt{settings['data']['filter']['min_nteam']}.ts{settings['data']['filter']['min_team_size']}" if filter else ""

    if exp_id: output = f'{output}exp_{exp_id}/'
    if not os.path.isdir(output): os.makedirs(output)

    datasets = {}
    models = {}

    if 'dblp' in domain_list: datasets['dblp'] = Publication
    if 'imdb' in domain_list: datasets['imdb'] = Movie
    if 'uspt' in domain_list: datasets['uspt'] = Patent
    if 'gith' in domain_list: datasets['gith'] = Repo

    # model names starting with 't' means that they will follow the streaming scenario
    # model names ending with _a1 means that they have one 1 added to their input for time as aspect learning
    # model names having _dt2v means that they learn the input embedding with doc2vec where input is (skills + year)

    # non-temporal (no streaming scenario, bag of teams)
    if 'random' in model_list: models['random'] = Rnd()
    if 'fnn' in model_list: models['fnn'] = Fnn()
    if 'bnn' in model_list: models['bnn'] = Bnn()
    if 'fnn_emb' in model_list: models['fnn_emb'] = Fnn()
    if 'bnn_emb' in model_list: models['bnn_emb'] = Bnn()
    if 'nmt' in model_list: models['nmt'] = Nmt()

    # streaming scenario (no vector for time)
    if 'tfnn' in model_list: models['tfnn'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn' in model_list: models['tbnn'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_emb' in model_list: models['tfnn_emb'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_emb' in model_list: models['tbnn_emb'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tnmt' in model_list: models['tnmt'] = tNmt(settings['model']['nfolds'], settings['model']['step_ahead'])

    # streaming scenario with adding one 1 to the input (time as aspect/vector for time)
    if 'tfnn_a1' in model_list: models['tfnn_a1'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_a1' in model_list: models['tbnn_a1'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tfnn_emb_a1' in model_list: models['tfnn_emb_a1'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_emb_a1' in model_list: models['tbnn_emb_a1'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])

    # streaming scenario with adding the year to the doc2vec training (temporal dense skill vecs in input)
    if 'tfnn_dt2v_emb' in model_list: models['tfnn_dt2v_emb'] = tNtf(Fnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])
    if 'tbnn_dt2v_emb' in model_list: models['tbnn_dt2v_emb'] = tNtf(Bnn(), settings['model']['nfolds'], step_ahead=settings['model']['step_ahead'])

    # todo: temporal: time as an input feature

    # temporal recommender systems
    if 'caser' in model_list: models['caser'] = Caser(settings['model']['step_ahead'])
    if 'rrn' in model_list: models['rrn'] = Rrn(settings['model']['baseline']['rrn']['with_zero'], settings['model']['step_ahead'])


    if 'np_ratio' in fair: settings['fair']['np_ratio'] = fair['np_ratio']
    if 'fairness' in fair: settings['fair']['fairness'] = fair['fairness']
    if 'k_max' in fair: settings['fair']['k_max'] = fair['k_max']
    if 'attribute' in fair: settings['fair']['attribute'] = fair['attribute']

    assert len(datasets) > 0
    assert len(datasets) == len(domain_list)
    assert len(models) > 0

    for (d_name, d_cls) in datasets.items():
        datapath = data_list[domain_list.index(d_name)]
        prep_output = f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}'
        vecs, indexes = d_cls.generate_sparse_vectors(datapath, f'{prep_output}{filter_str}', filter, settings['data'])
        gender = pd.read_csv(f'{prep_output}{filter_str}/females.csv', index_col=None)
        female_ids = gender['opentf_index'].values.tolist()
        vecs['gender'] = lil_matrix((1, vecs['member'].shape[1]))# as a single sparse vector 1 * |size of expert| whose nonzero indexes are the file indexes
        vecs['gender'][:, female_ids] = 1.
        vecs['all_female'] = lil_matrix((vecs['member'].shape[0], vecs['member'].shape[1]))
        vecs['all_female'][:, female_ids] = 1.
        
        if augment:
            female_in_team = vecs['member'].multiply(vecs['gender'])
            vecs['female'] = lil_matrix(female_in_team)
            with open(f'{prep_output}{filter_str}/teamsvecs.pkl', 'wb') as outfile: pickle.dump(vecs, outfile)

        # if augment:
        #     from tqdm import tqdm
        #     for row in tqdm(range(0, vecs['member'].shape[0])):
        #         # random_indices = random.sample([i for i in range(10)], 3)
        #         # random_indices = random.sample(female_ids, 30)
        #         random_indices = female_ids[:10]
        #         #vecs['member'][:, female_ids] = 1.0
        #         vecs['member'][row, random_indices] = 1.0

        year_idx = []
        for i in range(1, len(indexes['i2y'])):
            if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > settings['model']['nfolds']:
                year_idx.append(indexes['i2y'][i-1])
        year_idx.append(indexes['i2y'][-1])
        indexes['i2y'] = year_idx
        splits = create_evaluation_splits(vecs['id'].shape[0], settings['model']['nfolds'], settings['model']['train_test_split'], indexes['i2y'] if future else None, output=f'{prep_output}{filter_str}', step_ahead=settings['model']['step_ahead'])

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

            output_path = f"{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}/t{vecs_['skill'].shape[0]}.s{vecs_['skill'].shape[1]}.m{vecs_['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings['model']['baseline'][baseline_name].items() if v])}"
            if not os.path.isdir(output_path): os.makedirs(output_path)
            # copyfile('./param.py', f'{output_path}/param.py')
            # make_popular_and_nonpopular_matrix(vecs_, data_list[0])

            m_obj.run(splits, vecs_, indexes, f'{output_path}', settings['model']['baseline'][baseline_name], settings['model']['cmd'], None, None)
    if 'agg' in settings['model']['cmd']: aggregate(output)


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/toy.json)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str.lower, default=[], required=True, help='a list of domains; required; (eg. -domain dblp imdb uspt gith)')
    dataset.add_argument('-filter', type=int, default=0, choices=[0, 1], help='remove outliers? (e.g., -filter 0 (default) or 1)')
    dataset.add_argument('-future', type=int, default=0, choices=[0, 1], help='predict future? (e.g., -future 0 (default) or 1)')
    dataset.add_argument('-augment', type=int, default=0, choices=[0, 1], help='Augument data? (e.g., -augment 0 (default) or 1)')

    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-model', '--model-list', nargs='+', type=str.lower, default=[], required=True, help='a list of neural models (eg. -model random fnn bnn fnn_emb bnn_emb nmt)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: -output ./../output/)')
    output.add_argument('-exp_id', type=str, default=None, help='ID of the experiment')

    fair = parser.add_argument_group('fair')
    fair.add_argument('-np_ratio', '--np_ratio', type=float, default=None, required=False, help='desired ratio of non-popular experts after reranking; if None, based on distribution in dataset; default: None; Eg. 0.5')
    fair.add_argument('-fairness', '--fairness', nargs='+', type=str, default='det_greedy', required=False, help='reranking algorithm from {det_greedy, det_cons, det_relaxed}; required; Eg. det_cons')
    fair.add_argument('-k_max', '--k_max', type=int, default=None, required=False, help='cutoff for the reranking algorithms; default: None')
    fair.add_argument('-attribute', '--attribute', nargs='+' ,type=str, default='popularity', required=False, help='the set of our sensitive attributes')

# python -u main.py -data ../data/raw/dblp/toy.dblp.v12.json
# 						  ../data/raw/imdb/toy.title.basics.tsv
# 						  ../data/raw/uspt/toy.patent.tsv
#                         ../data/raw/gith/toy.data.csv
# 					-domain dblp imdb uspt gith
# 					-model random
# 					       fnn fnn_emb bnn bnn_emb nmt
# 					       tfnn tbnn tnmt tfnn_emb tbnn_emb tfnn_a1 tbnn_a1 tfnn_emb_a1 tbnn_emb_a1 tfnn_dt2v_emb tbnn_dt2v_emb
#                   -filter 1

# To run on compute canada servers you can use the following command: (time is in minutes)
#sbatch --account=def-hfani --mem=96000MB --time=2880 cc.sh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Team Formation')
    addargs(parser)
    args = parser.parse_args()

    fair = { 'fairness': args.fairness, 'k_max': args.k_max, 'np_ratio': args.np_ratio, 'attribute': args.attribute }
    run(data_list=args.data_list,
        domain_list=args.domain_list,
        fair = fair,
        filter=args.filter,
        future=args.future,
        augment=args.augment,
        model_list=args.model_list,
        output=args.output,
        exp_id=args.exp_id,
        settings=param.settings)

    # aggregate(args.output)

