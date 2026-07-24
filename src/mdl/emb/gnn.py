import os, itertools, pickle, logging, numpy as np, copy, time, re
log = logging.getLogger(__name__)

import pkgmgr as opentf
from mdl.earlystopping import EarlyStopping
from .t2v import T2v

class Gnn(T2v):

    def __init__(self, output, device, seed, cfg, model):
        super().__init__(output, device, seed, cfg, model)
        Gnn.torch = opentf.install_import('torch')
        Gnn.pyg = opentf.install_import('torch_geometric')
        opentf.set_seed(self.seed, Gnn.torch)
        self.writer = opentf.install_import('tensorboardX', from_module='SummaryWriter')
        self.w = None
        self.decoder = None

    def _prep(self, teamsvecs, splits=None, time_indexes=None):
        tqdm = opentf.install_import('tqdm', from_module='tqdm')
        file = self.output + f'/../{self.cfg.graph.structure[1]}.{self.cfg.graph.dup_edge if self.cfg.graph.dup_edge else "dup"}.graph.pkl'
        try:
            log.info(f'Loading graph of {tuple(self.cfg.graph.structure)} from {file}  ...')
            with open(file, 'rb') as infile: self.data = pickle.load(infile)
        except FileNotFoundError:
            # NOTE: for any change, unit test using https://github.com/fani-lab/OpeNTF/issues/280
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

    def learn(self, teamsvecs, splits=None, time_indexes=None):
        self._prep(teamsvecs)
        self.cfg.model = self.cfg[self.name] #gnn.n2v or gnn.gs --> gnn.model

        self.output += f'/{self.name}.b{self.cfg.model.b}.e{self.cfg.model.e}.ns{self.cfg.model.ns}.lr{self.cfg.model.lr}.es{self.cfg.model.es}.spe{self.cfg.model.spe}.d{self.cfg.model.d}.{self.cfg.graph.dup_edge}.{self.cfg.graph.structure[1]}'
        self.output += f'{".pre" if self.cfg.graph.pre else ""}'
        if self.name == 'n2v': self.output += f'.w{self.cfg.model.w}.wl{self.cfg.model.wl}.wn{self.cfg.model.wn}'
        elif self.name == 'm2v': self.output += f'.w{self.cfg.model.w}.wl{self.cfg.model.wl}.wn{self.cfg.model.wn}.{self.cfg.model.metapath_name[1]}'  # should be fixed
        elif self.name in {'gcn', 'gs', 'gat', 'gatv2', 'gin'}: self.output += f'.h{"-".join([str(i) for i in self.cfg.model.h]) if self.cfg.model.h and len(self.cfg.model.h) > 0 else None}.nn{"-".join([str(i) for i in self.cfg.model.nn])}'

        # replace the 1 dimensional node features with pretrained d2v skill vectors of required dimension
        if self.cfg.graph.pre: self._init_d2v_node_features(teamsvecs)

        log.info(f'{opentf.textcolor["blue"]}Training {self.name} {opentf.textcolor["reset"]}... ')

        # (1) transductive (for opentf, only edges matter)
        # -- all nodes 'skills', 'member', 'team' are seen
        # -- edges (a) all can be seen for message passing but valid/test edges are not for loss/supervision (common practice)
        #          (b) valid/test team-member edges are literally removed, may 'rarely' lead to disconnected member nodes (uncommon but very strict/pure, no leakage)
        #          (c) valid/test team-member and team-skill edges are literally removed, lead to disconnected team nodes, may be disconnected skills and members
        # for now, (1)(b)
        # (2) inductive
        # -- valid/test 'team' nodes are unseen >> future

        train_data = copy.deepcopy(self.data)
        # remove (member to team) and (team to member) edges whose teams are in test set
        test_teams_to_remove = Gnn.torch.tensor(splits['test'])
        mask = Gnn.torch.isin(train_data['member', 'to', 'team'].edge_index[1], test_teams_to_remove)
        tst_m2t_edges = train_data['member', 'to', 'team'].edge_index[:, mask]
        train_data['member', 'to', 'team'].edge_index = train_data['member', 'to', 'team'].edge_index[:, ~mask]
        # mask = Gnn.torch.isin(train_data['team', 'rev_to', 'member'].edge_index[0], test_teams_to_remove)
        # tst_t2m_edges = train_data['team', 'rev_to', 'member'].edge_index[:, mask]
        # train_data['team', 'rev_to', 'member'].edge_index = train_data['team', 'rev_to', 'member'].edge_index[:, ~mask]
        train_data['team', 'rev_to', 'member'].edge_index = train_data['member', 'to', 'team'].edge_index[[1, 0]] # or flip(0)

        train_data.validate(raise_on_error=True)

        ## homo test/valid construction for n2v and homo versions of gnns
        offsets = {}; offset = 0
        for node_type in train_data.node_types:
            offsets[node_type] = offset
            offset += train_data[node_type].num_nodes

        tst_member_homo = tst_m2t_edges[0] + offsets['member']
        tst_team_homo = tst_m2t_edges[1] + offsets['team']
        tst_m_t_edge_homo = Gnn.torch.stack([tst_member_homo, tst_team_homo], dim=0)

        # tst_team_homo = tst_t2m_edges[0] + offsets['team']
        # tst_member_homo = tst_t2m_edges[1] + offsets['member']
        # tst_t_m_edge_homo = Gnn.torch.stack([tst_team_homo, tst_member_homo], dim=0)
        # tst_t_m_edge_homo = tst_m_t_edge_homo[[1, 0]]

        tst_edge_homo = Gnn.torch.hstack([tst_m_t_edge_homo, tst_m_t_edge_homo[[1, 0]]])
        assert len(tst_edge_homo), f'{opentf.textcolor["red"]}Empty test member-team edge set!"{opentf.textcolor["reset"]}'

        for foldidx in splits['folds'].keys():
            fold_data = copy.deepcopy(train_data)
            # remove (member to team) and (team to member) edges whose teams are in valid set too
            valid_teams_to_remove = Gnn.torch.tensor(splits['folds'][foldidx]['valid'])
            v_m2t_mask = Gnn.torch.isin(fold_data['member', 'to', 'team'].edge_index[1], valid_teams_to_remove)
            val_m2t_edges = fold_data['member', 'to', 'team'].edge_index[:, v_m2t_mask]
            fold_data['member', 'to', 'team'].edge_index = fold_data['member', 'to', 'team'].edge_index[:, ~v_m2t_mask]

            # v_t2m_mask = Gnn.torch.isin(fold_data['team', 'rev_to', 'member'].edge_index[0], valid_teams_to_remove)
            # val_t2m_edges = fold_data['team', 'rev_to', 'member'].edge_index[:, v_t2m_mask]
            # fold_data['team', 'rev_to', 'member'].edge_index = fold_data['team', 'rev_to', 'member'].edge_index[:, ~v_t2m_mask]
            fold_data['team', 'rev_to', 'member'].edge_index = fold_data['member', 'to', 'team'].edge_index[[1, 0]]

            val_member_homo = val_m2t_edges[0] + offsets['member']
            val_team_homo = val_m2t_edges[1] + offsets['team']
            val_m_t_edge_homo = Gnn.torch.stack([val_member_homo, val_team_homo], dim=0)

            # val_team_homo = val_t2m_edges[0] + offsets['team']
            # val_member_homo = val_t2m_edges[1] + offsets['member']
            # val_t_m_edge_homo = Gnn.torch.stack([val_team_homo, val_member_homo], dim=0)
            # val_t_m_edge_homo = val_m_t_edge_homo[[1, 0]]

            # val_edge_homo = Gnn.torch.hstack([val_m_t_edge_homo, val_m_t_edge_homo[[1, 0]]])
            # assert len(val_edge_homo), f'{opentf.textcolor["red"]}Empty valid member-team edge set!"{opentf.textcolor["reset"]}'

            # we need this list to filter them from the negative sampling edges by pyg (hard neg sample edges)
            tst_val_edges_homo = Gnn.torch.hstack([val_edge_homo, tst_edge_homo])
            assert Gnn.torch.unique(tst_val_edges_homo.T, dim=0).size(0) == tst_val_edges_homo.T.size(0), f'{opentf.textcolor["red"]}Overlapping edges found in test and valid edge sets for homo version of the graph!"{opentf.textcolor["reset"]}'

            # random-walk-based (rw) including n2v and m2v, are unsupervised and learn node embeddings from scratch, using random initialization internally.
            # no need to manually create and initialize node embeddgins like in message-passing-based (mp) methods.
            # such models do not consume node attributes or features like mp methods do. they only use the graph structure.
            # the learned embeddings are inside an nn.Embedding layer that is initialized randomly and optimized during training.
            if self.name == 'n2v':
                # ImportError: 'Node2Vec' requires either the 'pyg-lib' or 'torch-cluster' package
                # install_import(f'torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-{self.torch.__version__}.html', 'torch_cluster')
                # import importlib; importlib.reload(self.pyg);importlib.reload(self.pyg.typing);importlib.reload(self.pyg.nn)
                self.model = self.pyg.nn.Node2Vec((fold_homo_data:=(fold_data.to_homogeneous(add_edge_type=True, add_node_type=True))).edge_index,
                                     num_nodes=fold_homo_data.num_nodes, #should be explicitly passed to accomodate possible isolated nodes
                                     embedding_dim=self.cfg.model.d,
                                     walk_length=self.cfg.model.wl,
                                     context_size=self.cfg.model.w,
                                     walks_per_node=self.cfg.model.wn,
                                     num_negative_samples=self.cfg.model.ns).to(self.device)

                # pos_edges_homo = Gnn.torch.hstack([fold_homo_data.edge_index, tst_val_edges_homo])
                # assert Gnn.torch.unique(pos_edges_homo.T, dim=0).size(0) == pos_edges_homo.size(0), f'{opentf.textcolor["red"]}Overlapping edges found in train, test, and valid edge sets for homo version of the graph!"{opentf.textcolor["reset"]}'

                self._train_rw(splits, foldidx, val_edge_homo, tst_edge_homo)
                # self._get_node_emb(homo_data=fold_homo_data) #logging purposes

            elif self.name == 'm2v':
                # assert isinstance(self.data, self.pyg.data.HeteroData), f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                assert len(self.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                self.model = self.pyg.nn.MetaPath2Vec(edge_index_dict=fold_data.edge_index_dict,
                                                      num_nodes_dict = {ntype: fold_data[ntype].num_nodes for ntype in fold_data.node_types}, #NOTE: if not explicitly set, it does num_nodes = int(edge_index[0].max()) + 1 !!
                                                      metapath=[tuple(mp) for mp in self.cfg.model.metapath_name[0]],
                                                      embedding_dim=self.cfg.model.d,
                                                      walk_length=self.cfg.model.wl,
                                                      context_size=self.cfg.model.w,
                                                      walks_per_node=self.cfg.model.wn,
                                                      num_negative_samples=self.cfg.model.ns).to(self.device)
                # m2v only creates embeddings for node types in metapaths, it skips for others, so
                # the node ids of original graph should be relative to m2v indexing
                val_m_t_edge_homo[0] = val_m2t_edges[0] + self.model.start['member']
                val_m_t_edge_homo[1] = val_m2t_edges[1] + self.model.start['team']
                val_edge_homo = Gnn.torch.hstack([val_m_t_edge_homo, val_m_t_edge_homo[[1, 0]]])
                tst_m_t_edge_homo[0] = tst_m2t_edges[0] + self.model.start['member']
                tst_m_t_edge_homo[1] = tst_m2t_edges[1] + self.model.start['team']
                tst_edge_homo = Gnn.torch.hstack([tst_m_t_edge_homo, tst_m_t_edge_homo[[1, 0]]])

                self._train_rw(splits, foldidx, val_edge_homo, tst_edge_homo)
                # self._get_node_emb(homo_data=fold_data) #logging purposes

            elif self.name == 'han':
                # assert isinstance(self.data, self.pyg.data.HeteroData), f'{opentf.textcolor["red"]}Hetero graph is needed for m2v. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                assert len(self.data.node_types) > 1, f'{opentf.textcolor["red"]}Hetero graph is needed for han. {self.cfg.graph.structure} is NOT hetero!{opentf.textcolor["reset"]}'
                raise NotImplementedError(f'{self.name} not integrated!')

            elif self.name == 'lant':
                # if self.name == 'lant': self.model.learn(self, self.cfg.model.e)  # built-in validation inside lant_encoder class
                raise NotImplementedError(f'{self.name} not integrated!')

            # message-passing-based >> default on homo, but can be wrapped into HeteroConv
            elif self.name in {'gcn', 'gs', 'gat', 'gatv2', 'gin'}:
                # by default, gnn methods are for homo data. We can wrap it by HeteroConv or manually simulate it >> I think Jamil did this >> future
                fold_train_homo_data = fold_data.to_homogeneous(add_edge_type=True, add_node_type=True)
                # building multilayer gnn-based model. Shouldn't depend on data. but as our graph has no features for node (for now), we need to assign a randomly initialized embeddings as node features.
                # so, we need the num_nodes of the graph
                self.model = self._built_model_mp(fold_train_homo_data.num_nodes).to(self.device)
                self._train_mp(splits, foldidx, fold_train_homo_data, val_edge_homo, tst_edge_homo)
                # self._get_node_emb(homo_data=fold_homo_data) #logging purposes


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

    def _sample_negatives_from_contexts(self, total_neg, train_homo_data, val_edge_homo, tst_edge_homo):
        """Split total_neg across type-contexts proportional to each type's positive-edge count."""

        def _build_neg_sampling_context(train_homo_data, val_edge_homo, tst_edge_homo):
            """
            Returns a list of per-type sampling contexts if supervision_edge_types is set,
            otherwise a single untyped context covering the whole graph.
            val/test edges are always folded into `forbidden` WITHOUT type filtering
            (forbidding extra pairs is always safe, even if their type differs from the context).
            """

            def _sorted_codes(edge_index_list, num_nodes):
                def _encode_edges(edge_index, num_nodes): return edge_index[0].long() * num_nodes + edge_index[1].long()

                codes = [_encode_edges(ei, num_nodes) for ei in edge_index_list if ei.numel() > 0]
                codes += [_encode_edges(ei.flip(0), num_nodes) for ei in edge_index_list if ei.numel() > 0]
                codes = Gnn.torch.cat(codes).unique()
                return codes.sort().values.to(self.device)

            if 'supervision_edge_types' in self.cfg.model and self.cfg.model.supervision_edge_types is not None:
                etype_to_id = {etype: i for i, etype in enumerate(self.data.edge_types)}
                ntype_to_id = {ntype: i for i, ntype in enumerate(self.data.node_types)}
                contexts = []
                for et in self.cfg.model.supervision_edge_types:
                    et = tuple(et)
                    et_id = etype_to_id[et]
                    src_type, _, dst_type = et
                    type_mask = (train_homo_data.edge_type == et_id)
                    pos_edge_index = train_homo_data.edge_index[:, type_mask.nonzero(as_tuple=True)[0]]

                    src_pool = (train_homo_data.node_type == ntype_to_id[src_type]).nonzero(as_tuple=True)[0].to(self.device)
                    dst_pool = (train_homo_data.node_type == ntype_to_id[dst_type]).nonzero(as_tuple=True)[0].to(self.device)

                    # val/test folded in as-is, no type filtering needed (over-forbidding is harmless)
                    forbidden = _sorted_codes([pos_edge_index, val_edge_homo, tst_edge_homo], train_homo_data.num_nodes)
                    contexts.append({'pos_edge_index': pos_edge_index, 'forbidden_sorted': forbidden, 'src_pool': src_pool, 'dst_pool': dst_pool, })
                return contexts
            else:
                forbidden = _sorted_codes([train_homo_data.edge_index, val_edge_homo, tst_edge_homo], train_homo_data.num_nodes)
                return [{'pos_edge_index': train_homo_data.edge_index, 'forbidden_sorted': forbidden, 'src_pool': None, 'dst_pool': None, }]

        def _sample_negative_edges(num_nodes, num_neg, forbidden_sorted, src_pool=None, dst_pool=None, oversample=1.3):
            """Untyped (src_pool/dst_pool=None -> any node) or typed (restricted pools) negative sampling."""
            if num_neg == 0: return Gnn.torch.empty((2, 0), dtype=Gnn.torch.long, device=self.device)
            out_src, out_dst, remaining = [], [], num_neg
            while remaining > 0:
                trial = int(remaining * oversample) + 1
                if src_pool is None: src = Gnn.torch.randint(0, num_nodes, (trial,), device=self.device)
                else: src = src_pool[Gnn.torch.randint(0, src_pool.numel(), (trial,), device=self.device)]
                if dst_pool is None: dst = Gnn.torch.randint(0, num_nodes, (trial,), device=self.device)
                else: dst = dst_pool[Gnn.torch.randint(0, dst_pool.numel(), (trial,), device=self.device)]
                keep = src != dst  # harmless no-op when pools are disjoint (typed, diff node types)
                src, dst = src[keep], dst[keep]
                codes = src * num_nodes + dst
                pos = Gnn.torch.searchsorted(forbidden_sorted, codes).clamp(max=forbidden_sorted.numel() - 1)
                is_forbidden = forbidden_sorted[pos] == codes
                src, dst = src[~is_forbidden], dst[~is_forbidden]
                take = min(src.numel(), remaining)
                out_src.append(src[:take])
                out_dst.append(dst[:take])
                remaining -= take
            return Gnn.torch.stack([Gnn.torch.cat(out_src), Gnn.torch.cat(out_dst)])

        # --- train negatives: typed (per supervision_edge_types) or untyped, filtered against val/test ---
        contexts = _build_neg_sampling_context(train_homo_data, val_edge_homo, tst_edge_homo)

        if total_neg == 0: return Gnn.torch.empty((2, 0), dtype=Gnn.torch.long, device=self.device)
        counts = [max(c['pos_edge_index'].size(1), 1) for c in contexts]
        total_pos = sum(counts)
        neg_edges = []
        for c, cnt in zip(contexts, counts):
            n_neg = int(round(total_neg * cnt / total_pos))
            neg_edges.append(_sample_negative_edges(train_homo_data.num_nodes, n_neg, c['forbidden_sorted'], src_pool=c['src_pool'], dst_pool=c['dst_pool']))
        return Gnn.torch.cat(neg_edges, dim=1)

    def _train_mp(self, splits, foldidx, train_homo_data, val_edge_homo, tst_edge_homo):
        try:
            log.info(f'Loading the model {self.output}/f{foldidx}.pt ...')
            return self.model.load_state_dict(Gnn.torch.load(f'{self.output}/f{foldidx}.pt', map_location=self.device)['model_state_dict'])
        except FileNotFoundError:
            log.info(f'{opentf.textcolor["yellow"]}File not found! Training ...{opentf.textcolor["reset"]}')

        pos_train_edge_homo = train_homo_data.edge_index
        if 'supervision_edge_types' in self.cfg.model and self.cfg.model.supervision_edge_types is not None:
            etype_to_id = {etype: i for i, etype in enumerate(self.data.edge_types)}
            sup_edge_ids = []
            for et in self.cfg.model.supervision_edge_types:
                et_id = etype_to_id[tuple(et)]
                mask = (train_homo_data.edge_type == et_id)
                sup_edge_ids.append(mask.nonzero(as_tuple=True)[0])
            sup_edge_ids = Gnn.torch.cat(sup_edge_ids, dim=0)
            pos_train_edge_homo = train_homo_data.edge_index[:, sup_edge_ids]

        num_train_neg = int(pos_train_edge_homo.size(1) * self.cfg.model.ns)
        neg_train_edge_homo = self._sample_negatives_from_contexts(num_train_neg, train_homo_data, val_edge_homo, tst_edge_homo)
        train_edge_label_index = Gnn.torch.cat([pos_train_edge_homo, neg_train_edge_homo], dim=1)
        train_edge_label = Gnn.torch.cat([Gnn.torch.ones(pos_train_edge_homo.size(1)), Gnn.torch.zeros(neg_train_edge_homo.size(1))])

        train_loader = self.pyg.loader.LinkNeighborLoader(data=train_homo_data, # the transductive part (all nodes but only train edges for message passing w/o test/valid edges. they are already removed.
                                                          edge_label_index=train_edge_label_index, # train edges are pos and neg (excluding test/valid) for loss calc
                                                          edge_label=train_edge_label,
                                                          num_neighbors=self.cfg.model.nn, # this should match the number of hops/layers
                                                          neg_sampling_ratio=0, # we need to explicitly implement typed (i.e., among the supervision_edge_types) negative sampling edges to avoid test/valid edges be part of them
                                                          batch_size=self.cfg.model.b, shuffle=True)
        valid_loader = self.pyg.loader.LinkNeighborLoader(data=train_homo_data,
                                                          edge_label_index=val_edge_homo, # only the member-team valid edges (pos) for valid loss calc based on valid set of each fold
                                                          edge_label=Gnn.torch.ones(val_edge_homo.size(1)),
                                                          num_neighbors=self.cfg.model.nn, # this should match the number of hops/layers
                                                          neg_sampling_ratio=0, # same logic as in test, we just want to predict existing member <-> team that are removed earlier. So, the train and valid loss are not on the same scale. Early stopping also works close to test logic.
                                                          batch_size=self.cfg.model.b, shuffle=False)

        # note that the same neg and pos train edges and valid loader (pos valid edges) are used over epochs. So, loss comparison between epochs is exactly for same pos and neg edges.
        if self.w is None: self.w = self.writer(log_dir=f'{self.output}/logs4tboard/run_{int(time.time())}')
        def _(e, loader, optimizer=None):
            if optimizer: self.model.train()
            else: self.model.eval()
            e_loss = 0
            for batch in loader:
                batch = batch.to(self.device)
                if optimizer: optimizer.zero_grad()
                # #the node indexes in batch.edge_index and batch.edge_label_index are localized in minibatches. batch.n_id contains the respective indexes passed in edge_label_index arg.
                # src_local, dst_local = batch.edge_label_index[0], batch.edge_label_index[1]
                # src_global, dst_global = batch.n_id[src_local], batch.n_id[dst_local]
                x = self.model.forward(batch.edge_index) # contains edges with neigbourhood sampling subset of train_homo_data.edge_index but w/o valid/test edges
                pred = self.model.decode(x[batch.edge_label_index[0]], x[batch.edge_label_index[1]])
                loss = self.torch.nn.functional.binary_cross_entropy_with_logits(pred, batch.edge_label.float(), reduction='mean')
                if optimizer: loss.backward(); optimizer.step();
                e_loss += loss.item()
                #this is just the embeddings of the nodes in the current batch, not all the node embeddings
                #better way is to render all skill node embeddings
                #self.writer.add_embedding(tag='node_emb' if optimizer else 'v_loss', mat=x, global_step=e)

            self.w.add_scalar(tag='t_loss' if optimizer else 'v_loss', scalar_value=e_loss, global_step=e)
            return (e_loss / len(loader)) if len(loader) > 0 else float('inf')

        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)
        earlystopping = EarlyStopping(Gnn.torch, patience=self.cfg.model.es, verbose=True, delta=self.cfg.model.lr, save_model=False, trace_func=log.info)
        self.torch.cuda.empty_cache()
        for e in range(self.cfg.model.e):
            log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["blue"]}Train Loss: {(t_loss:=_(e, train_loader, optimizer)):.4f}{opentf.textcolor["reset"]}')
            log.info(f'Fold {foldidx}/{len(splits["folds"]) - 1}, Epoch {e}, {opentf.textcolor["magenta"]}Valid Loss: {(v_loss:=_(e, valid_loader)):.4f}{opentf.textcolor["reset"]}')
            if self.cfg.model.spe and (e == 0 or ((e + 1) % self.cfg.model.spe) == 0):
                #self.model.eval()
                self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.e{e}.pt')
                log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.e{e}.pt')

            if earlystopping(v_loss, self.model).early_stop:
                log.info(f'Early stopping triggered at epoch: {e}')
                break
        #self.model.eval()
        self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.pt')
        log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.pt.')
        self.w.close()

    def _train_rw(self, splits, foldidx, val_edge_homo, tst_edge_homo):
        try:
            log.info(f'Loading the model {self.output}/f{foldidx}.pt ...')
            return self.model.load_state_dict(Gnn.torch.load(f'{self.output}/f{foldidx}.pt', map_location=self.device)['model_state_dict'])
        except FileNotFoundError:
            log.info(f'{opentf.textcolor["yellow"]}File not found! Training ...{opentf.textcolor["reset"]}')

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
            scores = (self.model.embedding.weight[val_edge_homo[0]] * self.model.embedding.weight[val_edge_homo[1]]).sum(dim=-1)

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

            if self.cfg.model.spe and (e == 0 or ((e + 1) % self.cfg.model.spe) == 0):
                # self.model.eval()
                self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.e{e}.pt')
                log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.e{e}.pt')

            scheduler.step(v_loss)
            if earlystopping(v_loss, self.model).early_stop:
                log.info(f'Early stopping triggered at epoch: {e}')
                break

        self.torch.save({'model_state_dict': self.model.state_dict(), 'cfg': self.cfg, 'f': foldidx, 'e': e, 't_loss': t_loss, 'v_loss': v_loss}, f'{self.output}/f{foldidx}.pt')
        log.info(f'{self.name} model with {opentf.cfg2str(self.cfg.model)} saved at {self.output}/f{foldidx}.pt')

    def _init_d2v_node_features(self, teamsvecs):
        flag = False
        log.info(f'Loading pretrained d2v embeddings {self.cfg.graph.pre} in {self.output} to initialize node features, or if not exist, train d2v embeddings from scratch ...')
        from .d2v import D2v
        d2v_cfg = opentf.str2cfg(self.cfg.graph.pre)
        d2v_cfg.embtype = self.cfg.graph.pre.split('.')[-1] # Check emb.d2v.D2v.train() for filename pattern
        d2v_cfg.lr = self.cfg.model.lr
        d2v_cfg.spe = self.cfg.model.spe
        # simple lazy load, or train from scratch if the file not found!
        d2v_obj = D2v(self.output, self.device, self.seed, d2v_cfg, 'd2v').learn(teamsvecs, time_indexes=None, splits=None)
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
            if homo_data is None: homo_data = self.data.to_homogeneous(add_edge_type=True, add_node_type=True)
            embeddings = self.model.embedding.weight.data.cpu() if self.name == 'n2v' else self.model(homo_data.edge_index.to(self.device)).detach().cpu()
            node_type_tensor = homo_data.node_type # tensor of shape [num_nodes]
            if node_type is not None: return embeddings[node_type_tensor == (self.data.node_types.index(node_type))]
            for i, node_type in enumerate(self.data.node_types):
                type_embeddings = embeddings[node_type_tensor == i]  # shape: [num_nodes_of_type, self.cfg.model.d]
                result[node_type] = type_embeddings
        return result

    def test(self, teamsvecs, splits, testcfg):
        # output should be consumable by the ntf.evaluate(), otherwise needs overriding
        assert os.path.isdir(self.output), f'{opentf.textcolor["red"]}No folder for {self.output} exist!{opentf.textcolor["reset"]}'
        log.info(f'{opentf.textcolor["blue"]}Testing {self.name} {opentf.textcolor["reset"]}... ')
        # our evaluation methodology is to prediction the connection of all experts/candidates to the node of a test team, so we only need the team indices
        tst_teams = Gnn.torch.as_tensor(splits['test'], device=self.device)
        experts = Gnn.torch.arange(self.data['member'].num_nodes, device=self.device)

        for foldidx in splits['folds'].keys():
            modelfiles = [f'{self.output}/f{foldidx}.pt']
            if testcfg.per_epoch: modelfiles += [f'{self.output}/{_}' for _ in os.listdir(self.output) if re.match(f'f{foldidx}.e\d+.pt', _)]
            for modelfile in sorted(sorted(modelfiles), key=len):
                self.model.load_state_dict(Gnn.torch.load(modelfile, map_location=self.device)['model_state_dict'])
                self.model.eval()

                for pred_set in (['test', 'train', 'valid'] if testcfg.on_train else ['test']):
                    if pred_set != 'test': raise NotImplementedError(f'Prediction on {pred_set} not integrated!')

                    z_experts, z_teams = self._get_node_emb(node_type='member'), self._get_node_emb(node_type='team')

                    Gnn.torch.cuda.empty_cache()
                    preds = []
                    with Gnn.torch.no_grad():
                        for t in tst_teams:
                            z_t = z_teams[t].unsqueeze(0).expand(len(experts), -1)
                            pred = Gnn.torch.sigmoid((z_t * z_experts).sum(dim=-1))
                            preds.append(pred.cpu())  # keep on CPU to save memory

                    match = re.search(r'(e\d+)\.pt$', os.path.basename(modelfile))
                    epoch = (match.group(1) + '.') if match else ''

                    preds = Gnn.torch.vstack(preds)
                    Gnn.torch.save({'y_pred': opentf.topk_sparse(Gnn.torch, preds, testcfg.topK) if (testcfg.topK and testcfg.topK < preds.shape[1]) else preds, 'uncertainty': None}, f'{self.output}/f{foldidx}.{pred_set}.{epoch}pred', pickle_protocol=4)
                    log.info(f'{self.name} model predictions for fold{foldidx}.{pred_set}.{epoch} has saved at {self.output}/f{foldidx}.{pred_set}.{epoch}pred')

    def evaluate(self, teamsvecs, splits, evalcfg):
        from mdl.ntf import Ntf
        ntfobj = Ntf(self.output, self.device, self.seed, None)
        ntfobj.output = self.output # to override the trailing class name ntf
        ntfobj.evaluate(teamsvecs, splits, evalcfg)

    def adila(self, teamsvecs, splits, faircfg):
        from mdl.ntf import Ntf
        ntfobj = Ntf(self.output, self.device, self.seed, None)
        ntfobj.output = self.output # to override the trailing class name ntf
        ntfobj.adila(teamsvecs, splits, faircfg)
