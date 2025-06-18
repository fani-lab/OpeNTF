import numpy as np, logging
log = logging.getLogger(__name__)

from .ntf import Ntf

class Rnd(Ntf):
    def __init__(self, output, device, pytorch, seed, cgf): super(Rnd, self).__init__(output, device, pytorch, seed, cgf)

    def init(self, input_size, output_size):
        class Model:
            def __init__(self, cfg, input_size, output_size): self.output_size = output_size
            def forward(self, x): return Ntf.torch.clamp(Ntf.torch.rand(self.output_size), min=1.e-6, max=1. - 1.e-6)
        self.model = Model(self.cfg, input_size, output_size)
        return self.model
    def learn(self, teamsvecs, splits, prev_model): self.init(-1, teamsvecs['member'].shape[1])
    def test(self, teamsvecs, splits, on_train=False, per_epoch=False):
        X_test = teamsvecs['skill'][splits['test'], :]
        y_test = teamsvecs['member'][splits['test']]
        test_dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X_test, y_test), batch_size=self.cfg.b, shuffle=False)

        for foldidx in splits['folds'].keys():
            y_pred = Ntf.torch.empty(0, teamsvecs['member'].shape[1])
            for X, y in test_dl:
                scores = self.model.forward(X)
                y_pred = np.vstack((y_pred, scores))
            Ntf.torch.save({}, f'{self.output}/f{foldidx}.pt', pickle_protocol=4)#dummy model save
            Ntf.torch.save({'y_pred': y_pred, 'uncertainty': None}, f'{self.output}/f{foldidx}.test.pred', pickle_protocol=4)
            log.info(f'{self.name()} model predictions for fold{foldidx}.test. has saved at {self.output}/f{foldidx}.test.pred')
