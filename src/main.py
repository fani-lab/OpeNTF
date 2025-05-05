import os,json
import argparse
import pickle
import random; random.seed(0)
import numpy as np; np.random.seed(0)
# import torch
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

# import pandas as pd
# from shutil import copyfile

# from sklearn.model_selection import KFold, train_test_split
# from scipy.sparse import lil_matrix

import hydra
from omegaconf import OmegaConf #,DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import get_class

from pkgmgr import install_import

# from cmn.tools import NumpyArrayEncoder, popular_nonpopular_ratio

# from cmn.tools import generate_popular_and_nonpopular

def get_splits(n_sample, n_folds, train_ratio=0.85, year_idx=None, output='./', step_ahead=1):
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

@hydra.main(version_base=None, config_path='.', config_name='config')
def run(cfg):
    if 'prep' in cfg.cmd:
        cfg.data.output += f'.mt{cfg.data.filter.min_nteam}.ts{cfg.data.filter.min_team_size}' if 'filter' in cfg.data and cfg.data.filter else ''
        if not os.path.isdir(cfg.data.output): os.makedirs(cfg.data.output)

        domain_cls = get_class(cfg.data.domain)

        # this will call the Team.generate_sparse_vectors(), which itself may (lazy) call Team.read_data(), which itself may (lazy) call {Publication|Movie|Repo|Patent}.read_data()
        vecs, indexes = domain_cls.gen_teamsvecs(cfg.data.source, cfg.data.output, cfg.data)

        #TODO? move this call for evaluation part?
        # skill coverage metric, all skills of each expert, all expert of each skills (supports of each skill, like in RarestFirst)
        vecs['skillcoverage'] = domain_cls.gen_skill_coverage(vecs, cfg.data.output) # after we have a sparse vector, we create es_vecs from that

        if 'embedding' in cfg.data and cfg.data.embedding.class_method:
            # Get command-line overrides for embedding.
            # Kinda tricky as we dynamically override a subconfig.
            # Use '+data.embedding.{...}=value' to override
            # Use '+data.embedding.{...}=null' to drop. The '~data.embedding.{...}' cannot be used here.
            emb_overrides = [o.replace('+data.embedding.', '') for o in HydraConfig.get().overrides.task if '+data.embedding.' in o]
            embcfg = OmegaConf.merge(OmegaConf.load('mdl/emb/config.yaml'), OmegaConf.from_dotlist(emb_overrides))
            OmegaConf.resolve(embcfg)
            cfg.data.embedding.config = embcfg
            cfg.data.embedding.config.model.gnn.pytorch = cfg.pytorch
            cls, method = cfg.data.embedding.class_method.split('_')
            cls = get_class(cls)
            t2v = cls(cfg.data.output, cfg.data.acceleration, cfg.data.embedding.config.model[cls.__name__.lower()])
            t2v.name = method
            t2v.train(vecs, indexes)
    if any(c in cfg.cmd for c in ['train', 'test', 'eval', 'plot']):
        year_idx = []
        for i in range(1, len(indexes['i2y'])): #e.g, [(0, 1900), (6, 1903), (14, 1906)] => the i shows the starting index for teams of the year
            if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > cfg.train.nfolds: year_idx.append(indexes['i2y'][i-1])
        year_idx.append(indexes['i2y'][-1])
        indexes['i2y'] = year_idx

        splits = get_splits(vecs['id'].shape[0], cfg.train.nfolds, cfg.train.train_test_split, indexes['i2y'] if future else None, output=f'{cfg.data.output}{filter_str}', step_ahead=cfg.train.step_ahead)

        # if a list, all see the exact splits of teams.
        # if individual, they see different teams in splits. But honestly, as we show the average results, no big deal, esp., as we do n-fold
        models = {}
        # model names t* will follow the streaming scenario
        # model names *_ts have timestamp (year) as a single added feature
        # model names *_ts2v learn temporal skill vectors via d2v when each doc is a stream of (skills: year of the team)

        # non-temporal (no streaming scenario, bag of teams)
        if 'random' in cfg.models: models['random'] = Rnd()
        if 'fnn' in model_list: models['fnn'] = Fnn()
        if 'bnn_old' in model_list: models['bnn_old'] = Bnn_Old()
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

        assert len(models) > 0

        # if the gnn embeddings exist or we need to generate random data
        if(args.emb_model):
            import torch

            # this string is needed to be appended to the output path of the final prediction results of fnn or bnn
            emb_settings_str = f'{args.emb_model}.{args.emb_graph_type}.undir.{args.emb_agg}.e{emb_e}.ns{emb_ns}.b{emb_b}.d{emb_d}'
            emb_filepath = f'{prep_output}{filter_str}/emb/{emb_settings_str}.emb.pt'

            # skip loading embedding or replacing emb with actual vecs if we only need evaluation
            if args.cmd != ['eval']:
                from scipy import sparse
                emb_skill = torch.load(emb_filepath, map_location=torch.device('cpu'))['skill'].detach().numpy()
                vecs['skill'] = sparse._lil.lil_matrix(torch.tensor(vecs['skill'] * emb_skill))


        for (m_name, m_obj) in models.items():
            vecs_ = vecs.copy()
            if m_name.find('_emb') > 0:
                if args.t2v_new: # new addition in the emb section for t2v
                    from gensim.models import Doc2Vec

                    args.t2v_w = 1 if args.t2v_w is None else args.t2v_w
                    args.t2v_dm = 1 if args.t2v_dm is None else args.t2v_dm

                    emb_settings_str = 'dt2v' if m_name.find('_dt2v') > 0 else 'skill' + f'.emb.d{args.t2v_d}.w{args.t2v_w}.dm{args.t2v_dm}' # e.g: skill.emb.d8.w1.dm1
                    emb_filepath = prep_output + f'/{args.t2v_model}/{emb_settings_str}.mdl'
                    t2v = Doc2Vec.load(emb_filepath)
                    vecs_['skill'] = t2v.dv.vectors
                    emb_settings_str = f'w2v.{emb_settings_str}' # for opentf output, we create a separate folder named as that : w2v.skill.emb.d8.w1.dm1 under fnn or bnn
                else:
                    t2v = Team2Vec(vecs, indexes, 'dt2v' if m_name.find('_dt2v') > 0 else 'skill', f'./../data/preprocessed/{d_name}/{os.path.split(datapath)[-1]}{filter_str}')
                    emb_setting = settings['model']['baseline']['emb']
                    t2v.train(emb_setting['d'], emb_setting['w'], emb_setting['dm'], emb_setting['e'])
                    vecs_['skill'] = t2v.dv()

            if m_name.endswith('a1'): vecs_['skill'] = lil_matrix(scipy.sparse.hstack((vecs_['skill'], lil_matrix(np.ones((vecs_['skill'].shape[0], 1))))))

            baseline_name = m_name.lstrip('t').replace('_emb', '').replace('_dt2v', '').replace('_a1', '')
            print(f'Running for (dataset, model): ({d_name}, {m_name}) ... ')

            if(args.emb_model or args.t2v_new):
                output_path = f"{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}/{emb_settings_str}/t{vecs_['skill'].shape[0]}.s{vecs_['skill'].shape[1] if args.emb_d is None else args.emb_d}.m{vecs_['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings['model']['baseline'][baseline_name].items() if v])}"
            else:
                output_path = f"{output}{os.path.split(datapath)[-1]}{filter_str}/{m_name}/t{vecs_['skill'].shape[0]}.s{vecs_['skill'].shape[1] if args.emb_d is None else args.emb_d}.m{vecs_['member'].shape[1]}.{'.'.join([k + str(v).replace(' ', '') for k, v in settings['model']['baseline'][baseline_name].items() if v])}"
            if not os.path.isdir(output_path): os.makedirs(output_path)
            copyfile('./param.py', f'{output_path}/param.py')
            # make_popular_and_nonpopular_matrix(vecs_, data_list[0])

            m_obj.run(splits, vecs_, indexes, f'{output_path}', settings['model']['baseline'][baseline_name], settings['model']['cmd'], settings['fair'], merge_skills=False)
    if 'agg' in cfg.cmd: aggregate(cfg.data.output)

# sample runs for different configs, including different prep, embeddings, model training, ..., are available as unit-test in
# ./github/workflows/*.yml

# To run on compute canada servers you can use the following command: (time is in minutes)
#sbatch --account=def-hfani --mem=96000MB --time=2880 computecanada.sh

if __name__ == '__main__': run()
