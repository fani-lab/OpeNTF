import os, re, numpy as np, logging, time

log = logging.getLogger(__name__)

import pkgmgr as opentf

from .fnn import Fnn
# these two only when curriculum learning
# from .tools import get_class_data_params_n_optimizer, adjust_learning_rate, apply_weight_decay_data_parameters
# from .superloss import SuperLoss

# from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class Bnn(Fnn):
    def __init__(self, output, pytorch, device, seed, cgf):
        super(Fnn, self).__init__(output, pytorch, device, seed, cgf)
        Fnn.btorch = opentf.install_import('bayesian-torch', 'bayesian_torch.models.dnn_to_bnn')

    def init(self, input_size, output_size):
        super().init(input_size, output_size)
        # these settings could be exposed to ./mdl/__config.yaml. More details https://github.com/IntelLabs/bayesian-torch
        const_bnn_prior_parameters = {
            'prior_mu': 0.0, 'prior_sigma': 1.0, 'posterior_mu_init': 0.0, 'posterior_rho_init': -3.0,
            'type': 'Flipout',  # Flipout or Reparameterization
            'moped_enable': False,  # True to initialize from the pretrained dnn weights
            'moped_delta': 0.5,
        }
        Fnn.btorch.dnn_to_bnn(self.model, const_bnn_prior_parameters)
        self.is_bayesian = True
        #future: partially/mix Bayesian: some layers are Bayesian, others are not, for a hybrid architectures.

    def test(self, model_path, splits, indexes, vecs, params, on_train_valid_set=False, per_epoch=False, merge_skills=False):

        if not os.path.isdir(model_path): raise Exception("The model does not exist!")
        # input_size = len(indexes['i2s'])
        input_size = vecs['skill'].shape[1]
        output_size = len(indexes['i2c'])

        X_test = vecs['skill'][splits['test'], :]
        y_test = vecs['member'][splits['test']]
        test_matrix = TFDataset(X_test, y_test)
        test_dl = DataLoader(test_matrix, batch_size=params['b'], shuffle=True, num_workers=0)

        for foldidx in splits['folds'].keys():
            modelfiles = [f'{model_path}/state_dict_model.f{foldidx}.pt']
            if per_epoch: modelfiles += [f'{model_path}/{_}' for _ in os.listdir(model_path) if
                                         re.match(f'state_dict_model.f{foldidx}.e\d+.pt', _)]

            for modelfile in modelfiles:
                self.init(input_size=input_size, output_size=output_size, param=params).to(self.device)
                self.load_state_dict(torch.load(modelfile))
                self.eval()

                for pred_set in (['test', 'train', 'valid'] if on_train_valid_set else ['test']):
                    if pred_set != 'test':
                        X = vecs['skill'][splits['folds'][foldidx][pred_set], :]
                        y = vecs['member'][splits['folds'][foldidx][pred_set]]
                        matrix = TFDataset(X, y)
                        dl = DataLoader(matrix, batch_size=params['b'], shuffle=True, num_workers=0)
                    else:
                        X = X_test; y = y_test; matrix = test_matrix
                        dl = test_dl

                    torch.cuda.empty_cache()
                    ####
                    # criterion = torch.nn.CrossEntropyLoss()
                    # criterion = self.cross_entropy(self, y_, y, ns, nns, unigram, weight)
                    y_pred = []
                    test_running_loss = 0.0

                    with torch.no_grad():
                        for batch_idx, (X, y) in enumerate(dl):
                            X = X.float().to(self.device)
                            # y = y.float().to(self.device)  # Ensure y_batch is long for CrossEntropyLoss

                            y_ = self(X)
                            # kl = get_kl_loss(self)
                            # ce_loss = criterion(y_, y)
                            # loss = ce_loss + kl / params['b']
                            # test_running_loss += loss.item()

                            y_pred_batch = y_.cpu().numpy()
                            y_pred.append(y_pred_batch)

                            # print(f'Fold {foldidx}, Batch {batch_idx}, {pred_set} set, Loss: {loss.item()}')
                    y_pred = np.vstack(y_pred).squeeze()

                    epoch = modelfile.split('.')[-2] + '.' if per_epoch else ''
                    epoch = epoch.replace(f'f{foldidx}.', '')
                    torch.save(y_pred, f'{model_path}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)