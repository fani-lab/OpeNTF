import numpy as np

from mdl.ntf import Ntf

class Rnd(Ntf):
    def __init__(self, pytorch): super(Rnd, self).__init__(pytorch)

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False, merge_skills=False):
        from mdl.cds import TFDataset

        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = TFDataset(X_test, y_test)
        test_dl = Ntf.torch.utils.data.DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

        for foldidx in splits['folds'].keys():
            Ntf.torch.cuda.empty_cache()
            with Ntf.torch.no_grad():
                y_pred = Ntf.torch.empty(0, test_dl.dataset.output.shape[1])
                for x, y in test_dl:
                    x = x.to(device=self.device)
                    scores = self.forward(None, y)#we need y to know the size of output
                    scores = scores.squeeze(1).cpu().numpy()
                    y_pred = np.vstack((y_pred, scores))
            Ntf.torch.save(self, f'{model_path}/state_dict_model.f{foldidx}.pt', pickle_protocol=4)#dummy model save
            Ntf.torch.save(y_pred, f'{model_path}/f{foldidx}.test.pred', pickle_protocol=4)

    def forward(self, x, y): return Ntf.torch.clamp(Ntf.torch.rand(y.shape), min=1.e-6, max=1. - 1.e-6)