import os, itertools, pickle, logging, numpy as np, copy, time
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.earlystopping import EarlyStopping
from .t2v import T2v

class Gnn(T2v):

    def __init__(self, output, device, name, cgf):
        super().__init__(output, device, name, cgf)
        Gnn.torch = opentf.install_import(cgf.pytorch, 'torch')
        Gnn.pyg = opentf.install_import(f'torch_geometric==2.6.1 torch_cluster==1.6.3 torch_sparse==0.6.18 torch_scatter==2.1.2 pyg_lib==0.4.0 -f https://data.pyg.org/whl/torch-{Gnn.torch.__version__}.html', 'torch_geometric')
        opentf.install_import('tensorboard==2.14.0', 'tensorboard')
        opentf.set_seed(self.cfg.seed, Gnn.torch)
        self.writer = opentf.install_import('tensorboardX==2.6.2.2', 'tensorboardX', 'SummaryWriter')
        self.w = None
        self.decoder = None

    def _prep(self, teamsvecs, indexes, splits):
        #NOTE: for any change, unit test using https://github.com/fani-lab/OpeNTF/issues/280
        # import numpy as np
        # from scipy.sparse import lil_matrix
        # teamsvecs = {}
        # teamsvecs['skill']=lil_matrix(np.array([[1,1,0],[1,0,1],[1,1,1]]))
        # teamsvecs['member']=lil_matrix(np.array([[1,0,1,0],[1,1,0,0],[0,1,0,1]]))
        # teamsvecs['loc']=lil_matrix(np.array([[1,0],[0,1],[1,0]]))
        tqdm = opentf.install_import('tqdm==4.65.0', 'tqdm', 'tqdm')
        file = self.output + f'/{self.cfg.graph.structure[1]}.{self.cfg.graph.dup_edge if self.cfg.graph.dup_edge else "dup"}.graph.pkl'
        try:
            log.info(f'Loading graph of {tuple(self.cfg.graph.structure)} from {file}  ...')
            with open(file, 'rb') as infile: self.data = pickle.load(infile)
            return self.data
        except FileNotFoundError:
            log.info(f'File not found! Constructing the graph of type {self.cfg.graph.structure[0]} ...')
            self.data = self.pyg.data.HeteroData()
            node_types = set()
            #edges
            for edge_type in self.cfg.graph.structure[0]:
                log.info(f'Adding edges of type {edge_type} ...')
                assert edge_type[0] in teamsvecs.keys() and teamsvecs[edge_type[0]] is not None, f'{opentf.textcolor["red"]}Teamsvecs do NOT have {edge_type[0]}{opentf.textcolor["reset"]}'
                teams = teamsvecs[edge_type[0]] # take one part of an edge from here
                edges = []
                for i, row1 in enumerate(tqdm(teams, total=teams.shape[0])):
                    row2 = teamsvecs[edge_type[2]][i] if edge_type[2] != 'team' else [i] # take the other part of the edge from here
                    # now add edges from all members of part 1 to part 2 (in this case, both are the same, so we take combinations of 2)
                    if edge_type[0] == edge_type[2]: edges += [t for t in itertools.combinations(row1.nonzero()[1], 2)]
                    # now add edges from all members of part 1 to part 2
                    else: edges += [t for t in itertools.product(row1.nonzero()[1], row2.nonzero()[1] if edge_type[2] != 'team' else row2)]

                self.data[tuple(edge_type)].edge_index = self.torch.tensor(edges, dtype=self.torch.long).t().contiguous()
                self.data[tuple(edge_type)].edge_attr = self.torch.tensor([1] * len(edges), dtype=self.torch.long)
                node_types = node_types.union({edge_type[0], edge_type[2]})
            #nodes
            for node_type in node_types: self.data[node_type].x = self.torch.tensor([[0]] * (teamsvecs[node_type].shape[1] if node_type != 'team' else teamsvecs['skill'].shape[0]), dtype=self.torch.float)

            # if not self.settings['dir']:
            log.info('To undirected graph ...')
            transform = self.pyg.transforms.ToUndirected(merge=False)
            self.data = transform(self.data)

            if self.cfg.graph.dup_edge:
                log.info(f'To merge duplicate edges by {self.cfg.graph.dup_edge} weights/features ...')
                transform = self.pyg.transforms.RemoveDuplicatedEdges(key=['edge_attr'], reduce=self.cfg.graph.dup_edge)
                self.data = transform(self.data)

            # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_self_loops

            self.data.validate(raise_on_error=True)
            log.info(f'Saving graph at {file} ...')
            with open(file, 'wb') as f: pickle.dump(self.data, f)
            return self.data

    def train(self, teamsvecs, indexes, splits):
        self._prep(teamsvecs, None, splits)
        self.cfg.model = self.cfg[self.name] #gnn.n2v or gnn.gs --> gnn.model
        self.output += f'/{self.name}.d{self.cfg.model.d}.e{self.cfg.model.e}.ns{self.cfg.model.ns}.{self.cfg.graph.dup_edge}.{self.cfg.graph.structure[1]}'
        self.output += f'{".pre" if self.cfg.graph.pre else ""}'
        # replace the 1 dimensional node features with pretrained d2v skill vectors of required dimension
        if self.cfg.graph.pre: self._init_d2v_node_features(teamsvecs, indexes, splits)

        log.info(f'{opentf.textcolor["blue"]}Training {self.name} {opentf.textcolor["reset"]}... ')
        train_data = copy.deepcopy(self.data)
        # (1) transductive (for opentf, only edges matter)
        # -- all nodes 'skills', 'member', 'team' are seen
        # -- edges (a) all can be seen for message passing but valid/test edges are not for loss/supervision (common practice)
        #          (b) valid/test team-member edges are literally removed, may 'rarely' lead to disconnected member nodes (uncommon but very strict/pure, no leakage)
        #          (c) valid/test team-member and team-skill edges are literally removed, lead to disconnected team nodes, may be disconnected skills and members
        # for now, (1)(b)
        # (2) inductive
        # -- valid/test 'team' nodes are unseen >> future

        # remove (member to team) and (team to member) edges whose teams are in test set
        test_teams_to_remove = Gnn.torch.tensor(splits['test'])
        mask = ~Gnn.torch.isin(train_data['member', 'to', 'team'].edge_index[1], test_teams_to_remove)
        train_data['member', 'to', 'team'].edge_index = train_data['member', 'to', 'team'].edge_index[:, mask]
        mask = ~Gnn.torch.isin(train_data['team', 'rev_to', 'member'].edge_index[0], test_teams_to_remove)
        train_data['team', 'rev_to', 'member'].edge_index = train_data['team', 'rev_to', 'member'].edge_index[:, mask]

        for foldidx in splits['folds'].keys():

            # remove (member to team) and (team to member) edges whose teams are in valid set too
            valid_teams_to_remove = Gnn.torch.tensor(splits['folds'][foldidx]['valid'])

            v_m2t_mask = Gnn.torch.isin(train_data['member', 'to', 'team'].edge_index[1], valid_teams_to_remove)
            val_m2t_edges = train_data['member', 'to', 'team'].edge_index[:, v_m2t_mask]
            train_data['member', 'to', 'team'].edge_index = train_data['member', 'to', 'team'].edge_index[:, ~v_m2t_mask]

            v_t2m_mask = Gnn.torch.isin(train_data['team', 'rev_to', 'member'].edge_index[0], valid_teams_to_remove)
            val_t2m_edges = train_data['team', 'rev_to', 'member'].edge_index[:, v_t2m_mask]
            train_data['team', 'rev_to', 'member'].edge_index = train_data['team', 'rev_to', 'member'].edge_index[:, ~v_t2m_mask]

            ## homo valid construction for n2v and homo versions of gnns
            offsets = {}; offset = 0
            for node_type in train_data.node_types:
                offsets[node_type] = offset
                offset += train_data[node_type].num_nodes

            val_member_homo = val_m2t_edges[0] + offsets['member']
            val_team_homo = val_m2t_edges[1] + offsets['team']
            # # same effect/view as above two lines when to_homo(). So, no need for below lines
            # val_team_homo = val_t2m_edges[0] + offsets['team']
            # val_member_homo = val_t2m_edges[1] + offsets['member']

            val_m_t_edge_index_homo = Gnn.torch.stack([val_member_homo, val_team_homo], dim=0)

            # random-walk-based (rw) including n2v and m2v, are unsupervised and learn node embeddings from scratch, using random initialization internally.
            # no need to manually create and initialize node embeddgins like in message-passing-based (mp) methods.
            # such models do not consume node attributes or features like mp methods do. they only use the graph structure.
            # the learned embeddings are inside an nn.Embedding layer that is initialized randomly and optimized during training.
            if self.name == 'n2v':
                if foldidx == 0: self.output += f'.w{self.cfg.model.w}.wl{self.cfg.model.wl}.wn{self.cfg.model.wn}'
                # ImportError: 'Node2Vec' requires either the 'pyg-lib' or 'torch-cluster' package
                # install_import(f'torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-{self.torch.__version__}.html', 'torch_cluster')
                # import importlib; importlib.reload(self.pyg);importlib.reload(self.pyg.typing);importlib.reload(self.pyg.nn)
                self.model = self.pyg.nn.Node2Vec((homo_data:=(train_data.to_homogeneous())).edge_index,
                                     embedding_dim=self.cfg.model.d,
                                     walk_length=self.cfg.model.wl,
                                     context_size=self.cfg.model.w,
                                     walks_per_node=self.cfg.model.wn,
                                     num_negative_samples=self.cfg.model.ns).to(self.device)

                self._train_rw(splits, foldidx, val_m_t_edge_index_homo)
                self._get_node_emb(homo_data=homo_data) #logging purposes

            elif self.name == 'm2v':
                # assert isinstance(self.data, self.pyg.data.HeteroData), f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                assert len(self.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                if foldidx == 0: self.output += f'.w{self.cfg.model.w}.wl{self.cfg.model.wl}.wn{self.cfg.model.wn}.{self.cfg.model.metapath_name[1]}' #should be fixed

                num_nodes_dict =

                self.model = self.pyg.nn.MetaPath2Vec(edge_index_dict=train_data.edge_index_dict,
                                                      num_nodes_dict = {ntype: train_data[ntype].num_nodes for ntype in train_data.node_types}, #NOTE: if not explicitly set, it does num_nodes = int(edge_index[0].max()) + 1 !!
                                                      metapath=[tuple(mp) for mp in self.cfg.model.metapath_name[0]],
                                                      embedding_dim=self.cfg.model.d,
                                                      walk_length=self.cfg.model.wl,
                                                      context_size=self.cfg.model.w,
                                                      walks_per_node=self.cfg.model.wn,
                                                      num_negative_samples=self.cfg.model.ns).to(self.device)
                self._train_rw(splits, foldidx, val_m_t_edge_index_homo)
                self._get_node_emb() #logging purposes

            elif self.name == 'han':
                # assert isinstance(self.data, self.pyg.data.HeteroData), f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                assert len(self.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for han. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                raise NotImplementedError(f'{self.name} not integrated!')

            elif self.name == 'lant':
                raise NotImplementedError(f'{self.name} not integrated!')

            # message-passing-based >> default on homo, but can be wrapped into HeteroConv
            elif self.name in {'gcn', 'gs', 'gat', 'gatv2', 'gin'}:
                output = f'.d{self.cfg.model.d}.e{self.cfg.model.e}.b{self.cfg.model.b}.lr{self.cfg.model.lr}.ns{self.cfg.model.ns}.h{"-".join([str(i) for i in self.cfg.model.h])}.nn{"-".join([str(i) for i in self.cfg.model.nn])}'

                # by default, gnn methods are for homo data. We can wrap it by HeteroConv or manually simulate it >> I think Jamil did this >> future
                homo_data = self.data.to_homogeneous()
                # building multilayer gnn-based model. Shouldn't depend on data. but as our graph has no features for node (for now), we need to assign a randomly initialized embeddings as node features.
                # so, we need the num_nodes of the graph
                self.model = self._built_model_mp(homo_data.num_nodes).to(self.device)
                train_l, valid_l, test_l = self._build_loader_mp(homo_data=homo_data) # building train/valid/test splits and loaders. Should depend on data
                self._train_mp(train_l, valid_l, test_l)

            # if self.name == 'lant': self.model.learn(self, self.cfg.model.e)  # built-in validation inside lant_encoder class
            #
            # self.plot_points()
        if self.w: self.w.close()

    def _built_model_mp(self, num_nodes):
        class Model(Gnn.torch.nn.Module):
            def __init__(self, cfg, name, num_nodes):
                super().__init__()
                Model.torch = Gnn.torch
                Model.pyg = Gnn.pyg

                self.node_emb = self.torch.nn.Embedding(num_nodes, cfg.model.d)
                self.torch.nn.init.xavier_uniform_(self.node_emb.weight)
                conv_cls = None
                if   name == 'gcn':   conv_cls = self.pyg.nn.GCNConv
                elif name == 'gs' :   conv_cls = self.pyg.nn.SAGEConv
                elif name == 'gat':   conv_cls = lambda in_ch, out_ch: self.pyg.nn.GATConv(in_ch, out_ch, heads=cfg.model.ah, concat=cfg.model.cat)
                elif name == 'gatv2': conv_cls = lambda in_ch, out_ch: self.pyg.nn.GATv2Conv(in_ch, out_ch, heads=cfg.model.ah, concat=cfg.model.cat)
                elif name == 'gin':   conv_cls = lambda in_ch, out_ch: self.pyg.nn.GINConv(self.torch.nn.Sequential(*[self.torch.nn.Linear(in_ch, out_ch), self.torch.nn.ReLU(), self.torch.nn.Linear(out_ch, out_ch)]))
                else: raise NotImplementedError(f'{name} not supported')

                self.encoder = self.torch.nn.ModuleList()
                if 'h' in cfg.model and cfg.model.h is not None and len(cfg.model.h) > 0:
                    for i, l in enumerate(cfg.model.h): self.encoder.append(conv_cls(cfg.model.d if i == 0 else cfg.model.h[i - 1], cfg.model.h[i]))
                else: self.encoder = self.torch.nn.ModuleList([conv_cls(cfg.model.d, cfg.model.d)])
            def forward(self, edge_index):
                x = self.node_emb.weight
                for conv in self.encoder: x = self.torch.nn.functional.relu(conv(x, edge_index))
                return x

            # decoder part: as simple as dot-product or as complex as a MLP-based binary classifier (indeed another end2end approach with fnn and bnn!)
            # decoder = torch.nn.Linear(hidden_dims[-1], 2)
            def decode(self, x_i, x_j): return (x_i * x_j).sum(dim=-1) # we use binary_cross_entropy_with_logits for loss calc
        return Model(self.cfg, self.name, num_nodes)
    def _build_loader_mp(self, homo_data):


        train_data, valid_data, test_data = self.pyg.transforms.RandomLinkSplit(is_undirected=True,
            num_val=0.1, num_test=0.0, # just for now. later, this should be based on the main splits of teams
            add_negative_train_samples=True, neg_sampling_ratio=self.cfg.model.ns)(homo_data)
        # for manually using a homo gnn method for hetero like gcn/gs, or a hetero gnn like heterogcn
        # transform = T.RandomLinkSplit(num_val=0.1, num_test=0.0, disjoint_train_ratio=0.3, neg_sampling_ratio=0.0,             # we leave negative sampling to the mini_batch_loaders
        #     add_negative_train_samples=False, , neg_sampling_ratio=self.cfg.model.ns,
        #     edge_types=edge_types, rev_edge_types=rev_edge_types)
        train_data.validate(raise_on_error=True); valid_data.validate(raise_on_error=True); test_data.validate(raise_on_error=True)

        # edge masks for the selected edge type in homogeneous graph
        if 'supervision_edge_types' in self.cfg.model and self.cfg.model.supervision_edge_types is not None:
            for t in self.cfg.model.supervision_edge_types:
                etype_id = train_data.edge_type_names.index('__'.join(self.cfg.model.supervision_edge_types))
                etype_mask = train_data.edge_type == etype_id
                train_edge_index = train_data.edge_index[:, etype_mask]

        train_loader = self.pyg.loader.LinkNeighborLoader(data=homo_data, # the transductive part: full graph for message passing
                                                         edge_label_index=train_data.edge_label_index, # the transductive part: only the train edges for loss calc
                                                         edge_label=train_data.edge_label,
                                                         num_neighbors=self.cfg.model.nn, # this should match the number of hops/layers
                                                         batch_size=self.cfg.model.b, shuffle=True)
        valid_loader = self.pyg.loader.LinkNeighborLoader(data=homo_data, # the transductive part: full graph for message passing
                                                         edge_label_index=valid_data.edge_label_index, # the transductive part: only the valid edges for loss calc
                                                         edge_label=valid_data.edge_label,
                                                         num_neighbors=self.cfg.model.nn, # this should match the number of hops/layers
                                                         batch_size=self.cfg.model.b, shuffle=False)
        test_loader = self.pyg.loader.LinkNeighborLoader(data=homo_data, # the transductive part: full graph for message passing
                                                         edge_label_index=test_data.edge_label_index, # the transductive part: only the test edges for loss calc
                                                         edge_label=test_data.edge_label,
                                                         num_neighbors=self.cfg.model.nn, # this should match the number of hops/layers
                                                         batch_size=self.cfg.model.b, shuffle=False)
        return train_loader, valid_loader, test_loader
        # the rest is for per edge type, per split loaders by jamil.
        # not sure we need per edge type loader
        # but per split is needed based on transductive/inductive train/valid/test teams

        #     #if self.model_name == 'han': self.metapath_name = self.settings['metapaths'][self.graph_type]
        #
        #     train_data, val_data, test_data, self.edge_types, self.rev_edge_types = self.split(self.data)
        #     # create separate loaders for separate seed edge_types
        #     self.train_loader, self.val_loader, self.test_loader = {}, {}, {}
        #     from torch_geometric.loader import LinkNeighborLoader
        #     for edge_type in self.cfg.graph.supervision_edge_types:
        #         # create a mbatch for a given split_data (mode = train / val / test)
        #         self.train_loader[edge_type] = LinkNeighborLoader(data=train_data, num_neighbors=self.cfg.model.nn, neg_sampling='binary',
        #                                                neg_sampling_ratio=self.cfg.model.ns,  # prev : neg_sampling = None
        #                                                edge_label_index=(edge_type, train_data[edge_type].edge_label_index),
        #                                                edge_label=train_data[edge_type].edge_label,
        #                                                batch_size=self.b , shuffle=True,)
        #
        #         self.val_loader[edge_type] = LinkNeighborLoader(data=val_data, num_neighbors=self.cfg.model.nn, neg_sampling='binary',
        #                                                neg_sampling_ratio=self.cfg.model.ns,  # prev : neg_sampling = None
        #                                                edge_label_index=(edge_type, val_data[edge_type].edge_label_index),
        #                                                edge_label=val_data[edge_type].edge_label,
        #                                                batch_size=3 * self.b , shuffle=True,)
        #
        #         # self.test_loader[edge_type] = self.create_mini_batch_loader(test_data, edge_type, 'test') # we dont need a test loader as of now
        #
        #     log.info(f'Device: {self.device}')
        #     self.torch.cuda.empty_cache()
        #     train_data.to(self.device)
        #     # the train_data is needed to collect info about the metadata
        #     from encoder import Encoder
        #     self.model = Encoder(hidden_channels=self.d, data=train_data, model_name=self.model).to(self.device)
        #     # if self.model_name == 'lant':
        #     #     from lant_encoder import Encoder
        #     #     model = Encoder(hidden_channels=self.d, data=train_data)
        #     self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
    def _train_mp(self, train_l, valid_l, test_l):
        def _(e, loader, optimizer=None):
            if optimizer: self.model.train()
            else: self.model.eval()
            e_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                if optimizer: optimizer.zero_grad()
                x = self.model.forward(batch.edge_index)
                pred = self.model.decode(x[batch.edge_label_index[0]], x[batch.edge_label_index[1]])
                loss = self.torch.nn.functional.binary_cross_entropy_with_logits(pred, batch.edge_label.float(), reduction='mean')
                if optimizer: loss.backward(); optimizer.step();
                e_loss += loss.item()
                #this is just the embeddings of the nodes in the current batch, not all the node embeddings
                #better way is to render the all skill node embeddings
                #self.writer.add_embedding(tag='node_emb' if optimizer else 'v_loss', mat=x, global_step=e)

            self.writer.add_scalar(tag='t_loss' if optimizer else 'v_loss', scalar_value=e_loss, global_step=e)

            return (e_loss / len(loader)) if len(loader) > 0 else float('inf')

        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
        earlystopping = EarlyStopping(Gnn.torch, patience=self.cfg.model.es, verbose=True, save_model=False, trace_func=log.info)
        self.torch.cuda.empty_cache()
        for e in range(self.cfg.model.e):
            log.info(f'Epoch {e}, {opentf.textcolor["blue"]}Train Loss: {(t_loss:=_(e, train_l, optimizer)):.4f}{opentf.textcolor["reset"]}')
            log.info(f'Epoch {e}, {opentf.textcolor["magenta"]}Valid Loss: {(v_loss:=_(e, valid_l)):.4f}{opentf.textcolor["reset"]}')
            if self.cfg.model.save_per_epoch:
                #self.model.eval()
                self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}.e{e}')
                log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}.e{e}')

            if earlystopping(v_loss, self.model).early_stop:
                log.info(f'Early stopping triggered at epoch: {e}')
                break
        log.info(f'{opentf.textcolor["yellow"]}Test Loss: {(tst_loss:=_(self.cfg.model.e, test_l)):.4f}')
        #self.model.eval()
        self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'e': e, 't_loss': t_loss, 'v_loss': v_loss, 'tst_loss': tst_loss}, self.output)
        log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}.')
        self.writer.close()

    def _train_rw(self, splits, foldidx, val_m_t_edge_index_homo):
        if self.w is None: self.w = self.writer(log_dir=f'{self.output}/logs4tboard/run_{int(time.time())}')
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
        scheduler = Gnn.torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)
        loader = self.model.loader(batch_size=self.cfg.model.b, shuffle=True)  # num_workers=os.cpu_count() not working in windows! also, cuda won't engage for the loader if num_workers param is passed
        earlystopping = EarlyStopping(Gnn.torch, patience=self.cfg.model.es, verbose=True, save_model=False, trace_func=log.info)
        self.torch.cuda.empty_cache()
        for e in range(self.cfg.model.e):
            t_loss = 0; self.model.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device)) #reduction is fixed to 'mean' internally
                loss.backward(); optimizer.step(); t_loss += loss.item()

            self.model.eval()
            scores = (self.model.embedding.weight[val_m_t_edge_index_homo[0]] * self.model.embedding.weight[val_m_t_edge_index_homo[1]]).sum(dim=-1)

            # w/ pos and neg samples for validation
            # pos_scores = (z[val_edge_index[0]] * z[val_edge_index[1]]).sum(dim=-1)
            # neg_edge_index = Gnn.pyg.utils.negative_sampling(edge_index=self.model.edge_index, num_nodes=self.model.num_nodes, num_neg_samples=val_edge_index.size(1), method='sparse')
            # neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
            # scores = Gnn.torch.cat([pos_scores, neg_scores])
            # labels = Gnn.torch.cat([self.torch.ones_like(pos_scores), Gnn.torch.zeros_like(neg_scores)])
            # v_loss = Gnn.torch.F.binary_cross_entropy_with_logits(scores, labels, reduction='mean').item()

            v_loss = Gnn.torch.nn.functional.binary_cross_entropy_with_logits(scores, self.torch.ones_like(scores), reduction='mean').item()

            t_loss /= len(loader); v_loss /= len(scores)
            # self.writer.add_embedding(node_embeddings, global_step=e) >> would be nice to see the convergence of embeddings for node
            self.w.add_scalar(tag=f'{foldidx}_t_loss', scalar_value=t_loss, global_step=e)
            self.w.add_scalar(tag=f'{foldidx}_v_loss', scalar_value=v_loss, global_step=e)
            log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["blue"]}Train Loss: {t_loss:.4f}{opentf.textcolor["reset"]}')
            log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["magenta"]}Valid Loss: {v_loss:.4f}{opentf.textcolor["reset"]}')

            if self.cfg.model.spe:
                # self.model.eval()
                self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.e{e}.pt')
                log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.e{e}.pt')

            scheduler.step(v_loss)
            if earlystopping(v_loss, self.model).early_stop:
                log.info(f'Early stopping triggered at epoch: {e}')
                break

        self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.pt')
        log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.pt')
    def _init_d2v_node_features(self, teamsvecs, indexes, splits):
        flag = False
        log.info(f'Loading pretrained d2v embeddings {self.cfg.graph.pre} in {self.output} to initialize node features, or if not exist, train d2v embeddings from scratch ...')
        from .d2v import D2v
        d2v_cfg = opentf.str2cfg(self.cfg.graph.pre)
        d2v_cfg.seed = self.cfg.seed
        d2v_cfg.embtype = self.cfg.graph.pre.split('.')[-1] # Check emb.d2v.D2v.train() for filename pattern
        d2v_cfg.lr = self.cfg.model.lr
        d2v_cfg.save_per_epoch = self.cfg.model.save_per_epoch
        # simple lazy load, or train from scratch if the file not found!
        d2v_obj = D2v(self.output, self.device, d2v_cfg).train(teamsvecs, indexes, splits)
        # the order is NOT correct in d2v, i.e., vecs[0] may be for vecs['s20']. Call D2v.natsortvecs(d2v_obj.model.wv)
        # d2v = Doc2Vec.load(self.cfg.graph.pre)
        for node_type in self.data.node_types:
            if node_type == 'team':
                # no issue as the docs are 0-based indexed 0 --> '0'
                assert d2v_obj.model.docvecs.vectors.shape[0] == teamsvecs['skill'].shape[0], f'{opentf.textcolor["red"]}Incorrect number of embeddings for teams!{opentf.textcolor["reset"]}'
                indices = [d2v_obj.model.docvecs.key_to_index[str(i)] for i in range(len(d2v_obj.model.docvecs))]
                assert np.allclose(d2v_obj.model.docvecs.vectors, d2v_obj.model.docvecs.vectors[indices]), f'{opentf.textcolor["red"]}Incorrect embedding for a team due to misorderings of embeddings!{opentf.textcolor["reset"]}'
                # assert np.array_equal(d2v_obj.model.docvecs.vectors, d2v_obj.model.docvecs.vectors[indices])
                # (d2v_obj.model.docvecs.vectors[2] == d2v_obj.model.docvecs['2']).all()
                self.data[node_type].x = self.torch.tensor(d2v_obj.model.docvecs.vectors); flag = True  # team vectors (dv) for 'team' nodes, else individual node vectors (wv)
            # either 'skill' or 'member', correct number of embeddings per skills xor members
            elif d2v_obj.model.wv.vectors.shape[0] == teamsvecs[node_type].shape[1]: self.data[node_type].x = self.torch.tensor(D2v.natsortvecs(d2v_obj.model.wv)); flag = True
        if d2v_obj.model.wv.vectors.shape[0] == teamsvecs['skill'].shape[1] + teamsvecs['member'].shape[1]:
            ordered_vecs = self.torch.tensor(D2v.natsortvecs(d2v_obj.model.wv))
            if 'member' in self.data.node_types: self.data['member'].x = ordered_vecs[:teamsvecs['member'].shape[1]] ;flag = True # the first part is all m*
            if 'skill' in self.data.node_types: self.data['skill'].x = ordered_vecs[teamsvecs['member'].shape[1]:]; flag = True  # the remaining is s*
        assert flag, f'{opentf.textcolor["red"]}Nodes features initialization with d2v embeddings NOT applied! Check the consistency of d2v {self.cfg.graph.pre} and graph node types {self.cfg.graph.structure}{opentf.textcolor["reset"]}'

    def get_dense_vecs(self, teamsvecs, vectype='skill'):
        if vectype in teamsvecs.keys(): return (teamsvecs[vectype] @ self._get_node_emb(node_type=vectype)) / teamsvecs[vectype].sum(axis=1) #average of selected embeddings, e.g., skillsubset of each teams
        return self._get_node_emb(node_type=vectype) #individual embeddings

    def _get_node_emb(self, homo_data=None, node_type=None):
        #NOTE: as the node indexes are exactly the skill, member, or team idx in teamsvecs, the embeddings are always aligned, i.e., s_i >> emb['skill'][i]
        # having a model, we always can have the embedding
        result = {}
        self.model.eval()
        if self.name == 'm2v':
            if node_type is not None:
                try: return self.model(node_type).detach().cpu()
                except KeyError as e: raise KeyError(f'{opentf.textcolor["yellow"]}No vectors for {node_type}.{opentf.textcolor["reset"]} Check if it is part of metapath -> {self.cfg.model.metapath_name}') from e
            for node_type in self.data.node_types: # self.model.start or self.model.end could be used for MetaPath2Vec model but ...
                try: result[node_type] = self.model(node_type).detach().cpu()
                except KeyError: log.warning(f'{opentf.textcolor["yellow"]}No vectors for {node_type}.{opentf.textcolor["reset"]} Check if it is part of metapath -> {self.cfg.model.metapath_name}' )
        else:
            # in n2v, the weights are indeed the embeddings, like w2v or d2v
            # in other models, self.model(self.data), that is the forward-pass produces the embedding
            if homo_data is None: homo_data = self.data.to_homogeneous()
            embeddings = self.model.embedding.weight.data.cpu() if self.name == 'n2v' else self.model(homo_data.edge_index.to(self.device)).detach().cpu()
            node_type_tensor = homo_data.node_type # tensor of shape [num_nodes]
            if node_type is not None: return embeddings[node_type_tensor == (self.data.node_types.index(node_type))]
            for i, node_type in enumerate(self.data.node_types):
                type_embeddings = embeddings[node_type_tensor == i]  # shape: [num_nodes_of_type, self.cfg.model.d]
                result[node_type] = type_embeddings
        return result

    def _train_mp_split(self): pass



    # def learn(self, loader, epochs):
    #     import torch.nn.functional as F
    #     #from mdl.earlystopping import EarlyStopping
    #     es = install_import('', 'mdl.earlystopping', 'EarlyStopping')
    #     epochs_taken = 0
    #     loss_array = []; val_auc_array = []; val_loss_array = []
    #     earlystopping = es(verbose=True, delta=0.001, path=f"{self.model_output}/state_dict_model.e{epochs}.pt", trace_func=log.info, save_model=False)
    #     for epoch in range(1, epochs + 1):
    #         self.optimizer.zero_grad()
    #         total_loss = 0; total_examples = 0
    #         self.torch.cuda.empty_cache()
    #         # train for loaders of all edge_types, e.g : train_loader['skill','to','team'], train_loader['member','to','team']
    #         for seed_edge_type in self.cfg.graph.supervision_edge_types:
    #             print(f'epoch {epoch:03d} : batching for train_loader for seed_edge_type : {seed_edge_type}')
    #             for sampled_data in loader[seed_edge_type]:
    #                 self.optimizer.zero_grad()
    #                 sampled_data.to(self.device)
    #                 pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #                 # The ground_truth and the pred shapes should be 1-dimensional
    #                 # we squeeze them after generation
    #                 if (type(sampled_data) == self.pyg.data.HeteroData): ground_truth = sampled_data[seed_edge_type].edge_label
    #                 else: ground_truth = sampled_data.edge_label
    #                 loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #                 loss.backward()
    #                 self.optimizer.step()
    #
    #                 total_loss += float(loss) * pred.numel()  # each batch might not contain same number of predictions, so we normalize it using the individual number of preds each turn
    #                 total_examples += pred.numel()
    #         avg_loss = (total_loss / total_examples)
    #         if (epoch % 10 == 0): log.info(f"\n.............Epoch: {epoch:03d}, Loss: {avg_loss:.6f}.............\n")
    #         loss_array.append(avg_loss)
    #
    #         self.optimizer.zero_grad()
    #         ls, auc = self.eval(self.val_loader)
    #         val_loss_array.append(ls)
    #         val_auc_array.append(auc)
    #
    #         epochs_taken += 1
    #         earlystopping(val_loss_array[-1], self.model)
    #         if earlystopping.early_stop:
    #             print(f"Early Stopping Triggered at epoch: {epoch}")
    #             break
    #
    #     # plot the figure and save
    #     fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.png'
    #     self.plot_graph(self.torch.arange(1, epochs_taken + 1, 1), loss_array, val_loss_array, fig_output=fig_output)
    #     fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.val_auc_per_epoch.png'
    #     self.plot_graph(self.torch.arange(1, epochs_taken + 1, 1), val_auc_array, xlabel='Epochs', ylabel='Val AUC', title=f'Validation AUC vs Epochs for Embedding Generation', fig_output=fig_output)
    #
    # def split(self, data, manual_neg_samples):
    #     edge_types = None; rev_edge_types = None
    #     if (type(data) == self.pyg.data.HeteroData):#from torch_geometric.data import HeteroData
    #         num_edge_types = len(data.edge_types)
    #         edge_types = data.edge_types[:num_edge_types // 2]
    #         rev_edge_types = data.edge_types[num_edge_types // 2:]
    #
    #     transform = T.RandomLinkSplit(num_val=0.1, num_test=0.0, disjoint_train_ratio=0.3,
    #         neg_sampling_ratio=0.0,             # we leave negative sampling to the mini_batch_loaders
    #         add_negative_train_samples=False, edge_types=edge_types, rev_edge_types=rev_edge_types,
    #     )
    #
    #     train_data, val_data, test_data = transform(data)
    #     if manual_neg_samples:
    #         for data_type in [train_data, val_data]: self.manual_negative_sampler(data_type, edge_types, self.ns)
    #     train_data.validate(raise_on_error=True); val_data.validate(raise_on_error=True); test_data.validate(raise_on_error=True)
    #
    #     # if han, add metapaths to the metadata
    #     if self.model_name == 'han':
    #         from torch_geometric.transforms import AddMetaPaths
    #         train_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False, drop_unconnected_node_types=False)(train_data)
    #         val_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False, drop_unconnected_node_types=False)(val_data)
    #
    #     return train_data, val_data, test_data, edge_types, rev_edge_types
    #
    # def manual_negative_sampler(self, data_type, edge_types, num_samples_ratio):
    #     from torch_geometric.utils import negative_sampling
    #     for edge_type in edge_types:
    #         pos_edge_index = data_type[edge_type].edge_label_index
    #         num_pos_samples = pos_edge_index.shape[1]
    #         num_neg_samples = int(num_samples_ratio * num_pos_samples)
    #
    #         neg_edge_index = negative_sampling( edge_index=data_type[edge_type].edge_index,
    #             num_nodes=(data_type[edge_type[0]].num_nodes,data_type[edge_type[2]].num_nodes),  # Specify number of nodes if necessary
    #             num_neg_samples=num_neg_samples, method='sparse'
    #         )
    #         # Concatenate positive and negative edges
    #         new_edge_label_index = self.torch.cat([pos_edge_index, neg_edge_index], dim=1)
    #         new_edge_label = self.torch.cat([self.torch.ones(num_pos_samples), self.torch.zeros(num_neg_samples)], dim=0)
    #
    #         data_type[edge_type].edge_label_index = new_edge_label_index
    #         data_type[edge_type].edge_label = new_edge_label
    #
    #     return data_type
    #
    # def eval(self, loader):
    #     import torch.nn.functional as F
    #     preds = []; ground_truths = []
    #     total_loss = 0; total_examples = 0
    #     self.model.eval()
    #     with self.torch.no_grad():
    #         for seed_edge_type in self.cfg.graph.supervision_edge_types:
    #             for sampled_data in loader[seed_edge_type]:
    #                 sampled_data.to(self.device)
    #                 tmp_pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #                 if (type(sampled_data) == self.pyg.data.HeteroData): # we have ground_truths per edge_label_index
    #                     tmp_ground_truth = sampled_data[seed_edge_type].edge_label
    #                 else: tmp_ground_truth = sampled_data.edge_label
    #
    #                 assert tmp_pred.shape == tmp_ground_truth.shape
    #                 loss = F.binary_cross_entropy_with_logits(tmp_pred, tmp_ground_truth)
    #                 total_loss += float(loss) * tmp_pred.numel()
    #                 total_examples += tmp_pred.numel()
    #
    #                 preds.append(tmp_pred)
    #                 ground_truths.append(tmp_ground_truth)
    #
    #     pred = self.torch.cat(preds, dim=0).cpu().numpy()
    #     ground_truth = self.torch.cat(ground_truths, dim=0).cpu().numpy()
    #     loss = total_loss / total_examples
    #     #from sklearn.metrics import roc_auc_score
    #     roc_auc_score = install_import('scikit-learn==1.2.2', 'sklearn.metrics', 'roc_auc_score')
    #     auc = roc_auc_score(ground_truth, pred)
    #     print()
    #     print(f'Val loss : {loss:.6f}')
    #     print(f"Val AUC: {auc:.6f}\n")
    #     # print(f'................... ending eval...................\n')
    #     return loss, auc
    #
    # # made specifically for the gnn training
    # def plot_graph(self, x, y, *args, xlabel='Epochs', ylabel='Loss', title='Loss vs Epochs', fig_output='plot.png'):
    #     from matplotlib import pyplot as plt
    #     plt.plot(x, y, label='Train')  # Plot the first set of data
    #
    #     if len(args) > 0:
    #         plt.plot(x, args[0], label='Valid')  # Plot the second set of data
    #         plt.legend()  # Add legend if there are two sets of data
    #
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     print(f'\nsaving figure as : {fig_output}\n')
    #     plt.savefig(fig_output)
    #     plt.clf()
    #     # plot.show()


# @torch.no_grad()
# def plot_points(colors):
#     model.eval()
#     z = model().cpu().numpy()
#     z = TSNE(n_components=2).fit_transform(z)
#     y = data.y.cpu().numpy()
#
#     plt.figure(figsize=(8, 8))
#     for i in range(dataset.num_classes):
#         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
#     # plt.scatter(z[:, 0], z[:, 1], s=20)
#     plt.axis('off')
#     plt.show()
#
#
# colors = [
#     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
# ]
# plot_points(colors)