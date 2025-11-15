import os, pickle, logging, numpy as np
log = logging.getLogger(__name__)

import hydra
from omegaconf import OmegaConf #,DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class

import pkgmgr as opentf

# from cmn.tools import popular_nonpopular_ratio, generate_popular_and_nonpopular

def get_splits(n_sample, n_folds, train_ratio, output, seed, year_idx=None, step_ahead=1):
    try:
        log.info(f'Loading splits from {output} ...')
        with open(output, 'rb') as f: splits = pickle.load(f)
        return splits
    except FileNotFoundError as e:
        scikit = opentf.install_import('scikit-learn', 'sklearn.model_selection')
        log.info(f'Splits file not found! Generating ...')
        if year_idx:
            train = np.arange(year_idx[0][0], year_idx[-step_ahead][0])  # for temporal folding, we do on each time interval ==> look at tntf.py
            test = np.arange(year_idx[-step_ahead][0], n_sample)
        else: train, test = scikit.train_test_split(np.arange(n_sample), train_size=train_ratio, random_state=seed, shuffle=True)

        splits = dict()
        splits['test'] = test
        splits['folds'] = dict()
        skf = scikit.KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
            splits['folds'][k] = dict()
            splits['folds'][k]['train'] = train[trainIdx]
            splits['folds'][k]['valid'] = train[validIdx]

        with open(output, 'wb') as f: pickle.dump(splits, f)
        return splits

def aggregate(output):
    import re
    pd = opentf.install_import('pandas')
    pattern = re.compile(r'(?<!f\d\.)test\.pred\.eval\.mean\.csv$')
    files = list()
    for dirpath, dirnames, filenames in os.walk(output): files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if pattern.search(file)]

    for row in files:
        if len(row) > 7: #to accomodate submodels in emb/transfer-based results
            row[-3], row[-2] = row[-3] + '@' + row[-2], row[-1]
            del row[-1]

    files = pd.DataFrame(files, columns=['', '', 'domain', 'dataset', 'split', 'model-setting', 'rfile'])
    rfiles = files.groupby('rfile')
    for rf, r in rfiles:
        dfff = pd.DataFrame()
        rsplits = r.groupby('split')
        for rs, rr in rsplits:
            names = []
            dff = pd.DataFrame()
            df = rsplits.get_group(rs)
            dfs = []
            log.info(f'{opentf.textcolor["green"]}{output}/ ... {opentf.textcolor["reset"]}')
            for i, row in df.iterrows():
                rfilename = f'{output}/{row["model-setting"]}/{rf}'.replace('@', '/')
                log.info(rfilename)
                df = pd.read_csv(rfilename, names=['metric', 'mean', 'std'], skiprows=1)
                df = df.set_index("metric")
                dfs.append(df)
                names += [row['model-setting'] + '-mean', row['model-setting'] + '-std']
            dfs = pd.concat(dfs, axis=1)
            dfs = dfs.set_axis(names, axis=1)
            dfs.to_csv(f"{output}/test.pred.eval.mean.agg.csv", index=True)
            log.info(f'{opentf.textcolor["green"]}Saved at {output}/test.pred.eval.mean.agg.csv. {opentf.textcolor["reset"]}')

@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg):
    t2v = None
    domain_cls = get_class(cfg.data.domain)

    cfg.data.output += f'.mt{cfg.data.filter.min_nteam}.ts{cfg.data.filter.min_team_size}' if 'filter' in cfg.data and cfg.data.filter else ''
    if not os.path.isdir(cfg.data.output): os.makedirs(cfg.data.output)

    # this will call the Team.generate_sparse_vectors(), which itself may (lazy) call Team.read_data(), which itself may (lazy) call {Publication|Movie|Repo|Patent}.read_data()
    teamsvecs, indexes = domain_cls.gen_teamsvecs(cfg.data.source, cfg.data.output, cfg.data)

    year_idx = []
    for i in range(1, len(indexes['i2y'])): #e.g, [(0, 1900), (6, 1903), (14, 1906)] => the i shows the starting index for teams of the year
        if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > cfg.train.nfolds: year_idx.append(indexes['i2y'][i-1])
    year_idx.append(indexes['i2y'][-1])
    indexes['i2y'] = year_idx

    splitstr = f'/splits.f{cfg.train.nfolds}.r{cfg.train.train_test_ratio}'
    splits = get_splits(teamsvecs['skill'].shape[0], cfg.train.nfolds, cfg.train.train_test_ratio, f'{cfg.data.output}{splitstr}.pkl', cfg.seed, indexes['i2y'] if cfg.train.step_ahead else None, step_ahead=cfg.train.step_ahead)

    # move this call for evaluation part?
    # skill coverage metric, all skills of each expert, all expert of each skills (supports of each skill, like in RarestFirst)
    teamsvecs['skillcoverage'] = domain_cls.gen_skill_coverage(teamsvecs, cfg.data.output, skipteams=splits['test'])
    cfg.data.output += splitstr #all models of anytype should be under a split strategy

    if 'embedding' in cfg.data and cfg.data.embedding.class_method:
        # Get command-line overrides for embedding. Kinda tricky as we dynamically override a subconfig.
        # Use '+data.embedding.{...}=value' to override
        # Use '+data.embedding.{...}=null' to drop. The '~data.embedding.{...}' cannot be used here.
        emb_overrides = [o.replace('+data.embedding.', '') for o in HydraConfig.get().overrides.task if '+data.embedding.' in o]
        embcfg = OmegaConf.merge(OmegaConf.load(cfg.data.embedding.config), OmegaConf.from_dotlist(emb_overrides))
        embcfg.model.spe = cfg.train.save_per_epoch
        OmegaConf.resolve(embcfg)
        cfg.data.embedding.config = embcfg
        cls, method = cfg.data.embedding.class_method.split('_') if cfg.data.embedding.class_method.find('_') else (cfg.data.embedding.class_method, None)
        cls = get_class(cls)
        #t2v = cls(cfg.data.output, cfg.data.acceleration, method, cfg.data.embedding.config.model[cls.__name__.lower()])
        t2v = cls(cfg.data.output, cfg.acceleration, cfg.seed, cfg.data.embedding.config.model[cls.__name__.lower()], method)
        t2v.learn(teamsvecs, splits)

    if cfg.cmd and any(c in cfg.cmd for c in ['train', 'test', 'eval']):

        # if a list, all see the exact splits of teams.
        # if individual, they see different teams in splits. But as we show the average results, no big deal, esp., as we do n-fold
        models = {}
        # model names t* will follow the streaming scenario
        # model names *_ts have timestamp (year) as a single added feature
        # model names *_ts2v learn temporal skill vectors via d2v when each doc is a stream of (skills: year of the team)
        # non-temporal (no streaming scenario, bag of teams)
        assert len(cfg.models.instances) > 0, f'{opentf.textcolor["red"]}No model instance for training! Check ./src/__config__.yaml and models.instances ... {opentf.textcolor["reset"]}'

        # Get command-line overrides for models. Kinda tricky as we dynamically override a subconfig.
        # Use '+models.{...}=value' to override
        # Use '+models.{...}=null' to drop
        mdl_overrides = [o.replace('+models.', '') for o in HydraConfig.get().overrides.task if '+models.' in o]
        mdlcfg = OmegaConf.merge(OmegaConf.load(cfg.models.config), OmegaConf.from_dotlist(mdl_overrides))
        mdlcfg.spe = cfg.train.save_per_epoch
        mdlcfg.tntf.tfolds = cfg.train.nfolds
        mdlcfg.tntf.step_ahead = cfg.train.step_ahead
        OmegaConf.resolve(mdlcfg)
        cfg.models.config = mdlcfg

        if cfg.train.merge_teams_w_same_skills: domain_cls.merge_teams_by_skills(teamsvecs, inplace=True)

        #  this way of injecting skill embeddings are not fold-based >> https://github.com/fani-lab/OpeNTF/issues/324
        #  the ntf models know the fold inside ntf.learn()
        #  but since the folding remove the team-member links, and embeddings are mainly used for skills (transfer-based)
        #  t2v.get_dense_vecs returns the vecs from t2v.model in the last fold (after going through the earlier folds)
        #  I think no big deal for the underlying fnn/bnn, all their folds use the skill embeddings based on graph based on last fold
        if 'embedding' in cfg.data and cfg.data.embedding.class_method:
            # t2v object knows the embedding method and ...
            skill_vecs = t2v.get_dense_vecs(teamsvecs, vectype='skill')
            assert skill_vecs.shape[0] == teamsvecs['skill'].shape[0], f'{opentf.textcolor["red"]}Incorrect number of embeddings for teams subset of skills!{opentf.textcolor["reset"]}'
            teamsvecs['original_skill'] = teamsvecs['skill'] #to accomodate skill_coverage metric and future use cases like in nmt
            teamsvecs['skill'] = skill_vecs

        for m in cfg.models.instances:
            cls_method = m.split('_')
            cls = get_class(cls_method[0]) # e.g., rnd, fnn, bnn, gnn, tNtf, ...
            output_ = (t2v.output if t2v else cfg.data.output)
            if cls_method[0] == 'mdl.emb.gnn.Gnn':
                assert t2v, f'{opentf.textcolor["red"]}The mdl.emb.gnn.Gnn instance needs a data.embedding.class_method! {opentf.textcolor["reset"]}'
                models[m] = t2v
            else: models[m] = cls(output_, cfg.acceleration, cfg.seed, cfg.models.config[cls.__name__.lower()])

            if len(cls_method) > 1: #for those wrappers that need internal model like in tNtf
                inner_cls = get_class(cls_method[1]) # e.g., rnd, fnn, bnn,
                models[m].model = inner_cls(output_, cfg.acceleration, cfg.seed, cfg.models.config[inner_cls.__name__.lower()])

            if 'train' in cfg.cmd:
                log.info(f'{opentf.textcolor["blue"]}Training team recommender instance {m} ... {opentf.textcolor["reset"]}')
                if cls_method[0] != 'mdl.emb.gnn.Gnn': models[m].learn(teamsvecs, splits, None)
                else: log.info(f'{opentf.textcolor["yellow"]}Training a Gnn instance is through data.embedding.class_method! {opentf.textcolor["reset"]}')

            if 'test'  in cfg.cmd:
                log.info(f'{opentf.textcolor["green"]}Testing team recommender instance {m} ... {opentf.textcolor["reset"]}')
                models[m].test(teamsvecs, splits, cfg.test)

            if 'eval'  in cfg.cmd:
                log.info(f'{opentf.textcolor["magenta"]}Evaluating team recommender instance {m} ... {opentf.textcolor["reset"]}')
                for key in cfg.eval.metrics: cfg.eval.metrics[key] = [m.replace('topk', cfg.eval.topk) for m in cfg.eval.metrics[key]]
                models[m].evaluate(teamsvecs, splits, cfg.eval)

            # make_popular_and_nonpopular_matrix(vecs_, data_list[0])

    # if 'fair' in cmd: self.fair(output, vecs, splits, fair_settings)

    log.info(f'{opentf.textcolor["green"]}Aggregating the test results from test.pred.eval.mean.csv files ... {opentf.textcolor["reset"]}')
    aggregate(cfg.data.output)

# sample runs for different configs, including different prep, embeddings, model training, ..., are available as unit-test in
# ./github/workflows/*.yml

# To run on compute canada servers you can use the following command: (time is in minutes)
#sbatch --account=def-hfani --mem=96000MB --time=2880 computecanada.sh

if __name__ == '__main__': run()
