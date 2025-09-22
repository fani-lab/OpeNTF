import logging
log = logging.getLogger(__name__)

import pkgmgr as opentf
from .fnn import Fnn
# these two only when curriculum learning
# from .tools import get_class_data_params_n_optimizer, adjust_learning_rate, apply_weight_decay_data_parameters
# from .superloss import SuperLoss

# from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

class Bnn(Fnn):
    def __init__(self, output, device, seed, cgf):
        super().__init__(output, device, seed, cgf)
        Fnn.btorch = opentf.install_import('bayesian-torch', 'bayesian_torch.models.dnn_to_bnn')

    def init(self, input_size, output_size):
        # these settings could be exposed to ./mdl/__config.yaml. More details https://github.com/IntelLabs/bayesian-torch
        const_bnn_prior_parameters = {
            'prior_mu': 0.0, 'prior_sigma': 1.0, 'posterior_mu_init': 0.0, 'posterior_rho_init': -3.0,
            'type': 'Flipout',  # Flipout or Reparameterization
            'moped_enable': False,  # True to initialize from the pretrained dnn weights
            'moped_delta': 0.5,
        }
        Fnn.btorch.dnn_to_bnn(super().init(input_size, output_size), const_bnn_prior_parameters)
        self.is_bayesian = True
        return self.model
        #future: partially/mix Bayesian: some layers are Bayesian, others are not, for a hybrid architectures.
