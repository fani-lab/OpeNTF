import os, pickle, re, time, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import KFold

from cmn.tools import NumpyArrayEncoder
from mdl.ntf import Ntf

class tNtf(Ntf):
    def __init__(self, model, tfold, step_ahead):
        super(tNtf, self).__init__()
        self.model = model
        self.tfold = tfold
        self.step_ahead = step_ahead #for now, only 1 step ahead

    def learn(self, splits, indexes, vecs, params, prev_model, output):
        year_idx = indexes['i2y']
        items_in_directory = os.listdir(output)
        if 'param.py' in items_in_directory: items_in_directory.remove('param.py')
        items_in_directory = [int(item) for item in items_in_directory]
        for i, v in enumerate(year_idx[:-self.step_ahead]):#the last years are for test.
            n_years_trained = len(items_in_directory)
            if n_years_trained > 1:
                print(f'The model has already been trained on year {min(items_in_directory)}')
                items_in_directory.remove(min(items_in_directory))        
                continue
            skf = KFold(n_splits=self.tfold, random_state=0, shuffle=True)
            train = np.arange(year_idx[i][0], year_idx[i + 1][0])
            for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
                splits['folds'][k]['train'] = train[trainIdx]
                splits['folds'][k]['valid'] = train[validIdx]

            output_ = f'{output}/{year_idx[i][1]}'
            if not os.path.isdir(output_): os.makedirs(output_)
            with open(f'{output_}/splits.json', 'w') as f: json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)

            self.model.learn(splits, indexes, vecs, params, prev_model, output_) #not recursive, but fine-tune over years
            prev_model = {foldidx: f'{output_}/state_dict_model.f{foldidx}.pt' for foldidx in splits['folds'].keys()}

    def run(self, splits, vecs, indexes, output, settings, cmd):
        on_train_valid_set = False  # random baseline cannot join this.
        per_instance = True
        per_epoch = False

        year_idx = indexes['i2y']
        output_ = f'{output}/{year_idx[-self.step_ahead - 1][1]}' #this folder will be created by the last model training

        if 'train' in cmd: self.learn(splits, indexes, vecs, settings, None, output)
        if 'test' in cmd:# todo: the prediction of each step ahead should be seperate
            # for i, v in enumerate(year_idx[-self.step_ahead:]):  # the last years are for test.
            #     tsplits['test'] = np.arange(year_idx[i][0], year_idx[i + 1][0] if i < len(year_idx) else len(indexes['i2t']))
            self.model.test(output_, splits, indexes, vecs, settings, on_train_valid_set, per_epoch)

        if 'eval' in cmd:# todo: the evaluation of each step ahead should be seperate
            self.model.evaluate(output_, splits, vecs, on_train_valid_set, per_instance, per_epoch)

        if 'plot' in cmd: self.model.plot_roc(output_, splits, on_train_valid_set)



