import os, json, numpy as np, logging, pickle
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.ntf import Ntf

class tNtf(Ntf):
    def __init__(self, model, tfold, step_ahead):
        super(tNtf, self).__init__(None)
        self.model = model
        self.tfold = tfold
        self.step_ahead = step_ahead #for now, only 1 step ahead

    def name(self): return self.__class__.__name__.lower() + '.' + self.model.__class__.__name__.lower()

    def learn(self, teamsvecs, indexes, splits, cfg, prev_model, output):
        year_idx = indexes['i2y']
        items_in_directory = os.listdir(output)
        items_in_directory = [int(item) for item in items_in_directory]

        scikit = opentf.install_import('scikit-learn==1.2.2', 'sklearn.model_selection')

        for i, v in enumerate(year_idx[:-self.step_ahead]):#the last years are for test.
            n_years_trained = len(items_in_directory)
            if n_years_trained > 1:
                log.info(f'The model has already been trained on year {min(items_in_directory)}')
                items_in_directory.remove(min(items_in_directory))        
                continue
            train = np.arange(year_idx[i][0], year_idx[i + 1][0])
            skf = scikit.KFold(n_splits=self.tfold, random_state=cfg.seed, shuffle=True)
            for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
                splits['folds'][k]['train'] = train[trainIdx]
                splits['folds'][k]['valid'] = train[validIdx]

            output_ = f'{output}/{year_idx[i][1]}'
            if not os.path.isdir(output_): os.makedirs(output_)
            with open(f'{output_}/splits.pkl', 'wb') as f: pickle.dump(splits, f)

            self.model.learn(teamsvecs, indexes, splits, cfg, prev_model, output_) #not recursive, but fine-tune over time intervals (years)
            prev_model = {foldidx: f'{output_}/state_dict_model.f{foldidx}.pt' for foldidx in splits['folds'].keys()}




