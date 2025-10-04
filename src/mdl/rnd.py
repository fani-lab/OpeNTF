import numpy as np, logging
log = logging.getLogger(__name__)

import pkgmgr as opentf
from .ntf import Ntf

class Rnd(Ntf):
    def __init__(self, output, device, seed, cfg): super(Rnd, self).__init__(output, device, seed, cfg)

    def test(self, teamsvecs, splits, testcfg):
        X_test = teamsvecs['skill'][splits['test'], :]
        y_test = teamsvecs['member'][splits['test']]
        test_dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X_test, y_test), batch_size=self.cfg.b, shuffle=False)
        for pred_set in (['test', 'train', 'valid'] if testcfg.on_train else ['test']):
            for foldidx in splits['folds'].keys():
                if pred_set != 'test':
                    X = teamsvecs['skill'][splits['folds'][foldidx][pred_set], :]
                    y = teamsvecs['member'][splits['folds'][foldidx][pred_set]]
                    dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X, y), batch_size=self.cfg.b, shuffle=False)
                else: dl = test_dl
                y_pred = []
                for X, y in dl: y_pred.append(Ntf.torch.clamp(Ntf.torch.rand(y.shape), min=1.e-6, max=1. - 1.e-6).squeeze(1))
                y_pred = Ntf.torch.vstack(y_pred)

                Ntf.torch.save({}, f'{self.output}/f{foldidx}.pt', pickle_protocol=4)#dummy model save
                Ntf.torch.save({'y_pred': opentf.topk_sparse(Ntf.torch, y_pred, testcfg.topK) if (testcfg.topK and testcfg.topK < y_pred.shape[1]) else y_pred, 'uncertainty': None}, f'{self.output}/f{foldidx}.{pred_set}.pred', pickle_protocol=4)
                epoch = '' #in a random model, there is no training nor per_epoch training
                log.info(f'{self.name()} model predictions for fold{foldidx}.{pred_set}.{epoch} has saved at {self.output}/f{foldidx}.{pred_set}.{epoch}pred')
