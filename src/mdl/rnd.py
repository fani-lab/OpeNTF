import numpy as np

import torch
from torch.utils.data import DataLoader

from mdl.ntf import Ntf
from mdl.cds import TFDataset

class Rnd(Ntf):
    def __init__(self):
        super(Rnd, self).__init__()

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False):
        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = TFDataset(X_test, y_test)
        test_dl = DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

        for foldidx in splits['folds'].keys():
            torch.cuda.empty_cache()
            with torch.no_grad():
                y_pred = torch.empty(0, test_dl.dataset.output.shape[1])
                for x, y in test_dl:
                    x = x.to(device=self.device)
                    scores = self.forward(None, y)#we need y to know the size of output
                    scores = scores.squeeze(1).cpu().numpy()
                    y_pred = np.vstack((y_pred, scores))
            torch.save(self, f'{model_path}/state_dict_model.f{foldidx}.pt', pickle_protocol=4)#dummy model save
            torch.save(y_pred, f'{model_path}/f{foldidx}.test.pred', pickle_protocol=4)

    def forward(self, x, y):
        x = torch.clamp(torch.rand(y.shape), min=1.e-6, max=1. - 1.e-6)
        return x