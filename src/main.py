import os, pickle, logging, numpy as np
log = logging.getLogger(__name__)

import hydra
from omegaconf import OmegaConf #,DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class

import pkgmgr as opentf

# from cmn.tools import popular_nonpopular_ratio, generate_popular_and_nonpopular

def get_splits(n_sample, n_folds, train_ratio, output, seed, year_idx=None, step_ahead=1):
    splitf = f'{output}/splits.pkl'
    try:
        log.info(f'Loading splits from {splitf} ...')
        with open(splitf, 'rb') as f: splits = pickle.load(f)
        return splits
    except FileNotFoundError as e:
        scikit = opentf.install_import('scikit-learn==1.2.2', 'sklearn.model_selection')
        log.info(f'Splits file not found! Generating ...')
        if year_idx:
            train = np.arange(year_idx[0][0], year_idx[-step_ahead][0])  # for temporal folding, we do on each time interval ==> look at tntf.py
            test = np.arange(year_idx[-step_ahead][0], n_sample)
        else:
            train, test = scikit.train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=seed, shuffle=True)

        splits = dict()
        splits['test'] = test
        splits['folds'] = dict()
        skf = scikit.KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
            splits['folds'][k] = dict()
            splits['folds'][k]['train'] = train[trainIdx]
            splits['folds'][k]['valid'] = train[validIdx]

        with open(splitf, 'wb') as f: pickle.dump(splits, f)
        return splits

def aggregate(output):
    import pandas as pd
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
    if any(c in cfg.cmd for c in ['prep', 'train']):
        cfg.data.output += f'.mt{cfg.data.filter.min_nteam}.ts{cfg.data.filter.min_team_size}' if 'filter' in cfg.data and cfg.data.filter else ''
        if not os.path.isdir(cfg.data.output): os.makedirs(cfg.data.output)

        domain_cls = get_class(cfg.data.domain)

        # this will call the Team.generate_sparse_vectors(), which itself may (lazy) call Team.read_data(), which itself may (lazy) call {Publication|Movie|Repo|Patent}.read_data()
        vecs, indexes = domain_cls.gen_teamsvecs(cfg.data.source, cfg.data.output, cfg.data)

        #TODO? move this call for evaluation part?
        # skill coverage metric, all skills of each expert, all expert of each skills (supports of each skill, like in RarestFirst)
        vecs['skillcoverage'] = domain_cls.gen_skill_coverage(vecs, cfg.data.output) # after we have a sparse vector, we create es_vecs from that

        year_idx = []
        for i in range(1, len(indexes['i2y'])): #e.g, [(0, 1900), (6, 1903), (14, 1906)] => the i shows the starting index for teams of the year
            if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > cfg.train.nfolds: year_idx.append(indexes['i2y'][i-1])
        year_idx.append(indexes['i2y'][-1])
        indexes['i2y'] = year_idx

        splits = get_splits(vecs['skill'].shape[0], cfg.train.nfolds, cfg.train.train_test_ratio, cfg.data.output, cfg.seed, indexes['i2y'] if cfg.train.step_ahead else None, step_ahead=cfg.train.step_ahead)

        if 'embedding' in cfg.data and cfg.data.embedding.class_method:
            # Get command-line overrides for embedding.
            # Kinda tricky as we dynamically override a subconfig.
            # Use '+data.embedding.{...}=value' to override
            # Use '+data.embedding.{...}=null' to drop. The '~data.embedding.{...}' cannot be used here.
            emb_overrides = [o.replace('+data.embedding.', '') for o in HydraConfig.get().overrides.task if '+data.embedding.' in o]
            embcfg = OmegaConf.merge(OmegaConf.load('mdl/emb/config.yaml'), OmegaConf.from_dotlist(emb_overrides))
            embcfg.model.seed = cfg.seed
            embcfg.model.gnn.pytorch = cfg.pytorch
            OmegaConf.resolve(embcfg)
            cfg.data.embedding.config = embcfg
            cls, method = cfg.data.embedding.class_method.split('_')
            cls = get_class(cls)
            t2v = cls(cfg.data.output, cfg.data.acceleration, cfg.data.embedding.config.model[cls.__name__.lower()])
            t2v.name = method
            t2v.train(vecs, indexes, splits)

    if any(c in cfg.cmd for c in ['train', 'test']):

        # if a list, all see the exact splits of teams.
        # if individual, they see different teams in splits. But as we show the average results, no big deal, esp., as we do n-fold
        models = {}
        # model names t* will follow the streaming scenario
        # model names *_ts have timestamp (year) as a single added feature
        # model names *_ts2v learn temporal skill vectors via d2v when each doc is a stream of (skills: year of the team)

        # non-temporal (no streaming scenario, bag of teams)

        for m in cfg.models.instances:
            import mdl # required for all models. Also, mdl/__init__.py should expose all submodules
            models[m] = eval(m, {'mdl': mdl})

        # # streaming scenario (no vector for time)
        # if 'tfnn' in model_list: models['tfnn'] = tNtf(Fnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tbnn' in model_list: models['tbnn'] = tNtf(Bnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tfnn_emb' in model_list: models['tfnn_emb'] = tNtf(Fnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tbnn_emb' in model_list: models['tbnn_emb'] = tNtf(Bnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tnmt' in model_list: models['tnmt'] = tNmt(cfg.train.nfolds, cfg.train.step_ahead)
        #
        # # streaming scenario with adding one 1 to the input (time as aspect/vector for time)
        # if 'tfnn_a1' in model_list: models['tfnn_a1'] = tNtf(Fnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tbnn_a1' in model_list: models['tbnn_a1'] = tNtf(Bnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tfnn_emb_a1' in model_list: models['tfnn_emb_a1'] = tNtf(Fnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tbnn_emb_a1' in model_list: models['tbnn_emb_a1'] = tNtf(Bnn(), cfg.train.nfolds, cfg.train.step_ahead)
        #
        # # streaming scenario with adding the year to the doc2vec training (temporal dense skill vecs in input)
        # if 'tfnn_dt2v_emb' in model_list: models['tfnn_dt2v_emb'] = tNtf(Fnn(), cfg.train.nfolds, cfg.train.step_ahead)
        # if 'tbnn_dt2v_emb' in model_list: models['tbnn_dt2v_emb'] = tNtf(Bnn(), cfg.train.nfolds, cfg.train.step_ahead)

        # # todo: temporal: time as an input feature
        #
        # # temporal recommender systems
        # if 'caser' in model_list: models['caser'] = Caser(settings['model']['step_ahead'])
        # if 'rrn' in model_list: models['rrn'] = Rrn(settings['model']['baseline']['rrn']['with_zero'], settings['model']['step_ahead'])

        assert len(models) > 0, f'{opentf.textcolor["red"]}No model instance for training! Check ./src/config.yaml and models.instances ... {opentf.textcolor["reset"]}'

        if 'embedding' in cfg.data and cfg.data.embedding.class_method:
            # t2v object knows the embedding method and ...
            vecs['skill'] = t2v.get_dense_vecs()['skill']

        from shutil import copyfile
        # from scipy.sparse import lil_matrix
        for (m_name, m_obj) in models.items():
            # if m_name.endswith('a1'): vecs_['skill'] = lil_matrix(scipy.sparse.hstack((vecs_['skill'], lil_matrix(np.ones((vecs_['skill'].shape[0], 1))))))
            log.info(f'Training team recommender instance {m_name} ... ')
            # find a way to show model-emb pair setting
            copyfile('./config.yaml', f'{cfg.data.output}/config.yaml')
            copyfile('./mdl/config.yaml', f'{cfg.data.output}/mdl/config.yaml')
            copyfile('./mdl/emb/config.yaml', f'{cfg.data.output}/mdl/emb/config.yaml')

            # make_popular_and_nonpopular_matrix(vecs_, data_list[0])

            m_obj.run(vecs, indexes, splits, f'{cfg.data.output}', cfg.cmd, cfg.models.instances[m_name])

    if 'agg' in cfg.cmd: aggregate(cfg.data.output)

# sample runs for different configs, including different prep, embeddings, model training, ..., are available as unit-test in
# ./github/workflows/*.yml

# To run on compute canada servers you can use the following command: (time is in minutes)
#sbatch --account=def-hfani --mem=96000MB --time=2880 computecanada.sh

if __name__ == '__main__': run()
