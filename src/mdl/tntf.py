import os, numpy as np, logging, pickle
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.ntf import Ntf

class tNtf(Ntf):
    def __init__(self, output, device, seed, cgf, model=None):
        super().__init__(output, device, seed, cgf)
        self.model = model

    def name(self): return self.__class__.__name__.lower() + '.' + self.model.__class__.__name__.lower() # e.g., tNtf.Rnd

    def learn(self, teamsvecs, indexes, splits, prev_model):
        year_idx = indexes['i2y']
        items_in_directory = [int(item) for item in os.listdir(self.model.output) if item.isdigit()]

        scikit = opentf.install_import('scikit-learn', 'sklearn.model_selection')

        for i, v in enumerate(year_idx[:-self.cfg.step_ahead]):#the last years are for test.
            n_years_trained = len(items_in_directory)
            if n_years_trained > 1:
                log.info(f'The model has already been trained on year {min(items_in_directory)}')
                items_in_directory.remove(min(items_in_directory))        
                continue
            train = np.arange(year_idx[i][0], year_idx[i + 1][0])
            skf = scikit.KFold(n_splits=self.cfg.tfolds, random_state=self.seed, shuffle=True)
            for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
                splits['folds'][k]['train'] = train[trainIdx]
                splits['folds'][k]['valid'] = train[validIdx]

            output_ = f'{self.model.output}/{year_idx[i][1]}' #e.g., ../output/dblp/toy.dblp.v12.json/tntf/fnn/2000 or ../output/dblp/toy.dblp.v12.json/d128.e100.d2v.w5.dm1.skill_/tntf/fnn/2000
            if not os.path.isdir(output_): os.makedirs(output_)
            with open(f'{output_}/splits.pkl', 'wb') as f: pickle.dump(splits, f)

            self.model.learn(teamsvecs, indexes, splits, prev_model) #not recursive, but fine-tune over time intervals (years)
            prev_model = {foldidx: f'{output_}/state_dict_model.f{foldidx}.pt' for foldidx in splits['folds'].keys()}




