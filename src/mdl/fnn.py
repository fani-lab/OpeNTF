import os, re, numpy as np, logging, time
log = logging.getLogger(__name__)

import pkgmgr as opentf

from .ntf import Ntf
from .earlystopping import EarlyStopping

# these two only when curriculum learning
# from .tools import get_class_data_params_n_optimizer, adjust_learning_rate, apply_weight_decay_data_parameters
# from .superloss import SuperLoss

class Fnn(Ntf):

    def init(self, input_size, output_size):
        class Model(Ntf.torch.nn.Module):
            def __init__(self, cfg, input_size, output_size):
                super().__init__()
                self.layers = Ntf.torch.nn.ModuleList()
                self.layers.append(Ntf.torch.nn.Linear(input_size, cfg.h[0]))
                for i in range(1, len(cfg.h)): self.layers.append(Ntf.torch.nn.Linear(cfg.h[i - 1], cfg.h[i]))
                self.layers.append(Ntf.torch.nn.Linear(cfg.h[-1], output_size))
                for m in self.layers: Ntf.torch.nn.init.xavier_uniform_(m.weight)
            def forward(self, x):
                for i, l in enumerate(self.layers): x = Ntf.torch.nn.functional.leaky_relu(l(x))
                return x
                #leave the sigmoid and clamping to be handled by the BCEWithLogitsLoss()
                #return Ntf.torch.clamp(Ntf.torch.sigmoid(x), min=1.e-6, max=1. - 1.e-6)
        self.model = Model(self.cfg, input_size, output_size)
        return self.model

    def bxe(self, y_, y):
        condition = (y == 1) # positive (correct) experts
        if self.cfg.nsd == 'uniform': topk_indices = self.ns_uniform(y) #upscale selected neg experts
        elif self.cfg.nsd == 'unigram': topk_indices = self.ns_unigram(y)
        elif self.cfg.nsd == 'unigram_b': topk_indices = self.ns_unigram_batch(y)
        # elif self.cfg.nsd == 'inverse_unigram': weight = self.ns_inverse_unigram(y_, y)
        # elif self.cfg.nsd == 'inverse_unigram_b': weight = self.ns_inverse_unigram_mini_batch(y_, y)
        if self.cfg.nsd:
            selected_mask = Ntf.torch.zeros_like(y, dtype=Ntf.torch.bool)
            batch_indices = Ntf.torch.arange(y.shape[0], device=y.device).unsqueeze(1).expand(-1, self.cfg.ns)  # shape [B, ns]
            selected_mask[batch_indices, topk_indices] = True
            condition = condition | selected_mask #those selected neg experts, same loss weight as pos expert

        weight = Ntf.torch.where(condition, self.cfg.tpw, self.cfg.tnw) # the rest neg experts. if this is 0, pure neg sampling
        return Ntf.torch.nn.functional.binary_cross_entropy_with_logits(y_, y, weight, reduction='none')

    def ns_uniform(self, y):
        # fully batch-wise and gpu-friendly
        mask_upweight_0s = (y == 0)
        rand_scores = Ntf.torch.rand_like(y, dtype=Ntf.torch.float)
        rand_scores = rand_scores.masked_fill(~mask_upweight_0s, -1.0)
        # for each row, get top-n random scores among negatives
        # if fewer than n negatives, topk will still return n, but may duplicate
        _ , topk_indices = Ntf.torch.topk(rand_scores, k=self.cfg.ns, dim=1)
        return topk_indices

    def ns_unigram(self, y):
        # fully batch-wise and gpu-friendly
        mask_upweight_0s = (y == 0)
        weight_matrix = self.unigram.expand(y.shape[0], -1) #self.unigram is already set in self.learn() or in self.ns_unigram_batch()
        sampling_weights = Ntf.torch.where(mask_upweight_0s, weight_matrix, Ntf.torch.zeros_like(weight_matrix))

        # RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
        # happens very rarely (like in toy imdb) for unigram_b where the frequencies of ALL neg experts are 0
        # so the final vector of prob weights are all zero
        row_sums = sampling_weights.sum(dim=1)
        uniform_weights = Ntf.torch.ones_like(sampling_weights) #fallback to uniform
        sampling_weights[row_sums == 0] = uniform_weights[row_sums == 0]

        topk_indices = Ntf.torch.multinomial(sampling_weights, self.cfg.ns, replacement=False)  # [batch, ns]
        return topk_indices

    def ns_unigram_batch(self, y):
        self.unigram = Ntf.torch.tensor(y.sum(axis=0) / y.shape[0]).to(self.device)  # frequency of each expert in a batch
        return self.ns_unigram(y)

    def learn(self, teamsvecs, splits, prev_model):
        input_size = teamsvecs['skill'].shape[1]
        output_size = teamsvecs['member'].shape[1]

        if self.cfg.nsd == 'unigram': self.unigram = Ntf.torch.tensor(teamsvecs['member'].sum(axis=0)/teamsvecs['member'].shape[0]).to(self.device) # frequency of each expert in all previous teams
        #for 'unigram_b', we can calculate it based on y or y_ as it's size is the size of the batch

        w = self.writer(log_dir=f'{self.output}/logs4tboard/run_{int(time.time())}')
        for foldidx in splits['folds'].keys():

            # TODO: to inject fold-based pretrained t2v embeddings for skills

            X_train = teamsvecs['skill'][splits['folds'][foldidx]['train'], :]
            y_train = teamsvecs['member'][splits['folds'][foldidx]['train']]
            X_valid = teamsvecs['skill'][splits['folds'][foldidx]['valid'], :]
            y_valid = teamsvecs['member'][splits['folds'][foldidx]['valid']]

            train_dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X_train, y_train), batch_size=self.cfg.b, shuffle=True)
            valid_dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X_valid, y_valid), batch_size=self.cfg.b, shuffle=False)
            data_loaders = {'train': train_dl, 'valid': valid_dl}

            # Initialize network
            self.init(input_size=input_size, output_size=output_size)
            if prev_model: self.model.load_state_dict(Ntf.torch.load(prev_model[foldidx], map_location=self.device))
            self.model.to(self.device)

            optimizer = Ntf.torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
            scheduler = Ntf.torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
            # scheduler = Ntf.torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

            # if self.cfg.l == 'cdp': class_parameters, optimizer_class_param = get_class_data_params_n_optimizer(nr_classes=y_train.shape[1], lr=self.cfg.lr, device=self.device)
            # if self.cfg.l == 'csl': csl_criterion = SuperLoss(nsamples=X_train.shape[0], ncls=y_train.shape[1], wd_cls=0.9, loss_func=Ntf.torch.nn.BCELoss())
            earlystopping = EarlyStopping(Ntf.torch, patience=self.cfg.es, verbose=True, delta=self.cfg.lr, save_model=False, trace_func=log.info)

            for e in range(self.cfg.e):
                # if self.cfg.l == 'cdp' and e in (learning_rate_schedule:=[2, 4, 10]): adjust_learning_rate(model_initial_lr=self.cfg.lr, optimizer=optimizer, gamma=0.1, step=np.sum(e >= learning_rate_schedule))

                t_loss = v_loss = 0.0
                # Each epoch has a training and validation phase
                for phase in ['train', 'valid']:
                    for batch_idx, (X, y) in enumerate(data_loaders[phase]):
                        Ntf.torch.cuda.empty_cache()
                        X = X.squeeze(1).to(self.device)
                        y = y.squeeze(1).to(self.device)
                        if phase == 'train':
                            self.model.train(True)  # scheduler.step()
                            optimizer.zero_grad(); #self.cfg.l == 'cdp' and optimizer_class_param.zero_grad()

                            y_ = self.model.forward(X)

                            # if self.cfg.l == 'csl': loss = csl_criterion(y_.squeeze(1), y.squeeze(1), index)
                            # elif self.cfg.l == 'cdp':
                            #     data_parameter_minibatch = Ntf.torch.exp(class_parameters).view(1, -1)
                            #     y_ = y_ / data_parameter_minibatch
                            #     loss = self.cross_entropy(y_, y)
                            #     cdp_loss = apply_weight_decay_data_parameters(loss, class_parameter_minibatch=class_parameters, weight_decay=0.9)
                            # else:
                            loss = self.bxe(y_, y).sum(dim=1).mean() #reduction: 'sum' per instance, 'mean' over batch due to sparsity of multi-hot expert vector in last layer
                            if self.is_bayesian: loss += Fnn.btorch.get_kl_loss(self.model) / y.shape[0]
                            loss.backward(); #shouldn't we have this: if self.cfg.l == 'cdp': cdp_loss.backward()
                            # clip_grad_value_(model.parameters(), 1)
                            optimizer.step(); #self.cfg.l == 'cdp' and optimizer_class_param.step()
                            t_loss += loss.item()

                        else:  # valid
                            self.model.eval()  # Set model to valid mode
                            with Ntf.torch.no_grad():
                                y_ = self.model.forward(X)
                                # if self.cfg.l == 'csl': csl_criterion(y_.squeeze(), y.squeeze(), index)
                                # else:
                                loss = self.bxe(y_, y).sum(dim=1).mean() #look at train loss for the reason
                                if self.is_bayesian: loss += Fnn.btorch.get_kl_loss(self.model) / y.shape[0]
                                #how about the loss of cdp for each class/expert? cdp_loss
                                v_loss += loss.item()

                t_loss /= len(train_dl); v_loss /= len(valid_dl)
                w.add_scalar(tag=f'{foldidx}_t_loss', scalar_value=t_loss, global_step=e)
                w.add_scalar(tag=f'{foldidx}_v_loss', scalar_value=v_loss, global_step=e)
                log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["blue"]}Train Loss: {t_loss:.4f}{opentf.textcolor["reset"]}')
                log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["magenta"]}Valid Loss: {v_loss:.4f}{opentf.textcolor["reset"]}')
                if self.cfg.spe and (e == 0 or ((e + 1) % self.cfg.spe) == 0):
                    # self.model.eval()
                    self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.e{e}.pt')
                    log.info(f'{self.name()} model with {opentf.cfg2str(self.cfg)} saved at {self.output}/f{foldidx}.e{e}.pt')

                scheduler.step(v_loss)
                if earlystopping(v_loss, self.model).early_stop:
                    log.info(f'Early stopping triggered at epoch: {e}')
                    break

            self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.pt')
            log.info(f'{self.name()} model with {opentf.cfg2str(self.cfg)} saved at {self.output}/f{foldidx}.pt')
        w.close()

    def test(self, teamsvecs, splits, testcfg):
        assert os.path.isdir(self.output), f'{opentf.textcolor["red"]}No folder for {self.output} exist!{opentf.textcolor["reset"]}'
        input_size = teamsvecs['skill'].shape[1]
        output_size = teamsvecs['member'].shape[1]

        X_test = teamsvecs['skill'][splits['test'], :]
        y_test = teamsvecs['member'][splits['test']]
        test_dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X_test, y_test), batch_size=self.cfg.b, shuffle=False)

        for foldidx in splits['folds'].keys():
            modelfiles = [f'{self.output}/f{foldidx}.pt']
            if testcfg.per_epoch: modelfiles += [f'{self.output}/{_}' for _ in os.listdir(self.output) if re.match(f'f{foldidx}.e\d+.pt', _)]

            for modelfile in sorted(sorted(modelfiles), key=len):
                self.init(input_size=input_size, output_size=output_size).to(self.device)
                self.model.load_state_dict(Ntf.torch.load(modelfile, map_location=self.device)['model_state_dict'])
                self.model.eval()

                for pred_set in (['test', 'train', 'valid'] if testcfg.on_train else ['test']):
                    if pred_set != 'test':
                        X = teamsvecs['skill'][splits['folds'][foldidx][pred_set], :]
                        y = teamsvecs['member'][splits['folds'][foldidx][pred_set]]
                        dl = Ntf.torch.utils.data.DataLoader(Ntf.dataset(X, y), batch_size=self.cfg.b, shuffle=False)
                    else: dl = test_dl

                    Ntf.torch.cuda.empty_cache()
                    with Ntf.torch.no_grad():
                        y_pred = []
                        for XX, yy in dl:
                            XX = XX.squeeze(1).to(self.device)
                            if self.is_bayesian:
                                output_mc = []; pred_uncertainty = []; model_uncertainty = []
                                for mc_run in range(self.cfg.nmc): output_mc.append(Ntf.torch.nn.functional.sigmoid(self.model.forward(XX))) #model returns logits
                                output = Ntf.torch.stack(output_mc)
                                butil = opentf.install_import('bayesian-torch', 'bayesian_torch.utils.util')
                                pred_uncertainty.append(butil.predictive_entropy(output.data.cpu().numpy()))
                                model_uncertainty.append(butil.mutual_information(output.data.cpu().numpy()))
                                y_pred.append(output.mean(dim=0).cpu())

                            else: y_pred.append((Ntf.torch.nn.functional.sigmoid(self.model.forward(XX)).squeeze(1)).cpu())
                            # move each batch to main memory before appending; gpu may not have enough memory for the entire test set, like in the dblp dataset
                        y_pred = Ntf.torch.vstack(y_pred)

                    match = re.search(r'(e\d+)\.pt$', os.path.basename(modelfile))
                    epoch = (match.group(1) + '.') if match else ''

                    Ntf.torch.save({'y_pred': opentf.topk_sparse(Ntf.torch, y_pred, testcfg.topK) if (testcfg.topK and testcfg.topK < y_pred.shape[1]) else y_pred, 'uncertainty': {'pred': pred_uncertainty, 'model': model_uncertainty} if self.is_bayesian else None}, f'{self.output}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)
                    log.info(f'{self.name()} model predictions for fold{foldidx}.{pred_set}.{epoch} has saved at {self.output}/f{foldidx}.{pred_set}.{epoch}pred')
