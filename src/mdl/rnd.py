import numpy as np

from .ntf import Ntf

class Rnd(Ntf):
    def __init__(self, output, device, pytorch, seed, cgf): super(Rnd, self).__init__(output, device, pytorch, seed, cgf)

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False, merge_skills=False):
        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = Ntf.dataset(X_test, y_test)
        test_dl = Ntf.torch.utils.data.DataLoader(test_matrix, batch_size=self.cfg.b, shuffle=True)# num_workers=os.cpu_count() not working in windows! also, cuda won't engage for the loader if num_workers param is passed

        for foldidx in splits['folds'].keys():
            Ntf.torch.cuda.empty_cache()
            with Ntf.torch.no_grad():
                y_pred = Ntf.torch.empty(0, test_dl.dataset.output.shape[1])
                for x, y in test_dl:
                    scores = self.forward(None, y)#we need y to know the size of output
                    scores = scores.squeeze(1).cpu().numpy()
                    y_pred = np.vstack((y_pred, scores))
            Ntf.torch.save(self, f'{model_path}/state_dict_model.f{foldidx}.pt', pickle_protocol=4)#dummy model save
            Ntf.torch.save(y_pred, f'{model_path}/f{foldidx}.test.pred', pickle_protocol=4)

    def forward(self, x, y): return Ntf.torch.clamp(Ntf.torch.rand(y.shape), min=1.e-6, max=1. - 1.e-6)