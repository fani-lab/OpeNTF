import os, re, numpy as np, logging, time, copy
log = logging.getLogger(__name__)

import pkgmgr as opentf

from .ntf import Ntf

# TODO: the gnn models need to inherit ntf to act like them, let's follow the aggregaion pattern, a single obj of T2v model
# node2vec, metapath2vec, homo versions (gcn, gs, gat, gatv2, gin)
# - test >> test edges predictions based on the gnn model of each fold
# - eval >> Ntf.evaluate()
class Gtf(Ntf):
    def __init__(self, output, device, seed, cgf, model=None):
        super().__init__(output, device, seed, cgf)
        self.model = model

    def learn(self, teamsvecs, splits, prev_model):
        self.model.learn(teamsvecs, splits)
    def test(self, teamsvecs, splits, testcfg):
        # we don't need it in t2v but we may add it there so we call self.t2v.test()
        # or implement here, which makes more sense as t2v is just for embedding
        # output should be consumable by the ntf.evaluate(), otherwise needs overriding
        assert os.path.isdir(self.model.output), f'{opentf.textcolor["red"]}No folder for {self.output} exist!{opentf.textcolor["reset"]}'
        self.model._prep(teamsvecs)
        self.model.cfg.model = self.model.cfg[self.name] #gnn.n2v or gnn.gs --> gnn.model
        self.model.output += f'/{self.model.name}.d{self.model.cfg.model.d}.e{self.model.cfg.model.e}.ns{self.model.cfg.model.ns}.{self.model.cfg.graph.dup_edge}.{self.model.cfg.graph.structure[1]}'
        self.model.output += f'{".pre" if self.model.cfg.graph.pre else ""}'
        log.info(f'{opentf.textcolor["blue"]}Testing {self.model.name} {opentf.textcolor["reset"]}... ')

        # our evaluation methodology is to prediction the connection of all experts/candidates to the node of a test team, so we only need the team indices
        tst_teams = Ntf.torch.as_tensor(splits['test'], device=self.device)
        experts = Ntf.torch.as_tensor(self.model.data['member'].num_nodes, device=self.device)
        ## homo test construction for n2v and homo versions of gnns
        offsets = {}; offset = 0
        for node_type in self.model.data.node_types:
            offsets[node_type] = offset
            offset += self.model.data[node_type].num_nodes
        tst_teams = tst_teams + offsets['team']

        for foldidx in splits['folds'].keys():
            if foldidx == 0:
                if self.model.name == 'n2v': self.output += f'.w{self.model.cfg.model.w}.wl{self.model.cfg.model.wl}.wn{self.model.cfg.model.wn}'
                elif self.model.name == 'm2v':
                    assert len(self.model.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.model.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                    self.output += f'.w{self.model.cfg.model.w}.wl{self.model.cfg.model.wl}.wn{self.model.cfg.model.wn}.{self.model.cfg.model.metapath_name[1]}'  # should be fixed
                    tst_teams = tst_teams - offsets['team'] + self.model.model.start['team']
                elif self.model.name == 'han':
                    assert len(self.model.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for han. {self.model.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                    raise NotImplementedError(f'{self.model.name} not integrated!')
                elif self.model.name == 'lant': raise NotImplementedError(f'{self.model.name} not integrated!')
                elif self.model.name in {'gcn', 'gs', 'gat', 'gatv2', 'gin'}: self.model.output += f'.d{self.model.cfg.model.d}.e{self.model.cfg.model.e}.b{self.model.cfg.model.b}.lr{self.model.cfg.model.lr}.ns{self.model.cfg.model.ns}.h{"-".join([str(i) for i in self.model.cfg.model.h])}.nn{"-".join([str(i) for i in self.model.cfg.model.nn])}'

            modelfiles = [f'{self.output}/f{foldidx}.pt']
            if testcfg.per_epoch: modelfiles += [f'{self.output}/{_}' for _ in os.listdir(self.output) if re.match(f'f{foldidx}.e\d+.pt', _)]
            for modelfile in sorted(sorted(modelfiles), key=len):
                self.model.model.load_state_dict(Ntf.torch.load(modelfile, map_location=self.device)['model_state_dict'])
                self.model.model.eval()

                for pred_set in (['test', 'train', 'valid'] if testcfg.on_train else ['test']):
                    if pred_set != 'test': raise NotImplementedError(f'Prediction on {pred_set} not integrated!')
                    if self.model.name in {'n2v', 'm2v'}: z = self.model.model(Ntf.torch.arange(self.model.data.num_nodes, device=self.device))
                    elif self.model.name in {'gcn', 'gs', 'gat', 'gatv2', 'gin'}: z = self.model.model(self.model.data.x.to(self.device), self.model.data.edge_index.to(self.device))

                    Ntf.torch.cuda.empty_cache()
                    with Ntf.torch.no_grad():
                        for t in tst_teams:
                            t_embed = z[t].unsqueeze(0).expand(len(experts), -1)
                            m_embed = z[experts]
                            pred = Ntf.torch.sigmoid((t_embed * m_embed).sum(dim=-1))
                            preds.append(pred.cpu())  # keep on CPU to save memory

                    match = re.search(r'(e\d+)\.pt$', os.path.basename(modelfile))
                    epoch = (match.group(1) + '.') if match else ''

                    preds = Ntf.torch.cat(preds, dim=0)
                    Ntf.torch.save({'y_pred': Ntf.to_topk_sparse(preds, testcfg.topK) if (testcfg.topK and testcfg.topK < preds.shape[1]) else preds, 'uncertainty': None}, f'{self.output}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)
                    log.info(f'{self.model.name()} model predictions for fold{foldidx}.{pred_set}.{epoch} has saved at {self.model.output}/f{foldidx}.{pred_set}.{epoch}pred')

    # ideally, no need for this if self.test() is nicely implemented
    # def evaluate(self, teamsvecs, splits, evalcfg):
    #     pass
