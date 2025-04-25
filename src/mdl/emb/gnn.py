import os, itertools, pickle, logging, numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

from pkgmgr import install_import, cfg2str, str2cfg
from .t2v import T2v

class Gnn(T2v):

    def __init__(self, output, device, cgf):
        super().__init__(output, device, cgf)
        self.name = 'n2v' #default model
        Gnn.torch = install_import(cgf.pytorch, 'torch')
        Gnn.pyg = install_import('torch_geometric==2.6.1', 'torch_geometric')

        self.loader = None
        self.optimizer = None

    def _prep(self, teamsvecs, indexes):
        #NOTE: for any change, unit test using https://github.com/fani-lab/OpeNTF/issues/280
        # import numpy as np
        # from scipy.sparse import lil_matrix
        # teamsvecs = {}
        # teamsvecs['skill']=lil_matrix(np.array([[1,1,0],[1,0,1],[1,1,1]]))
        # teamsvecs['member']=lil_matrix(np.array([[1,0,1,0],[1,1,0,0],[0,1,0,1]]))
        # teamsvecs['loc']=lil_matrix(np.array([[1,0],[0,1],[1,0]]))

        file = self.output + f'/{self.cfg.graph.structure[1]}.{self.cfg.graph.dup_edge if self.cfg.graph.dup_edge else "dup"}.graph.pkl'
        try:
            log.info(f'Loading graph of {tuple(self.cfg.graph.structure)} from {file}  ...')
            with open(file, 'rb') as infile: self.data = pickle.load(infile)
            return self.data
        except FileNotFoundError:
            log.info(f'File not found! Constructing the graph ...')

            if isinstance(self.cfg.graph.structure[0], str):#homo
                log.info(f'Creating a homo graph with {self.cfg.graph.structure[0]} node type ...')
                teams = teamsvecs[self.cfg.graph.structure[0]] #TODO: if node_type == 'team'
                edges = []
                for i, row in enumerate(tqdm(teams, total=teams.shape[0])):
                    for t in itertools.combinations(row.nonzero()[1], 2): edges += [t]
                edge_index = self.torch.tensor(edges, dtype=self.torch.long).t().contiguous() #[1,2][1,3][2,4] >> [1,1,2][2,3,4]
                nodes = self.torch.tensor([[0]] * teams.shape[1], dtype=self.torch.float)
                self.data = self.pyg.data.Data(x=nodes, edge_index=edge_index, edge_attr=self.torch.tensor([1] * len(edges), dtype=self.torch.long))
            else:
                log.info(f'Creating a hetero graph of type {self.cfg.graph.structure[0]} ...')
                self.data = self.pyg.data.HeteroData()
                node_types = set()
                #edges
                for edge_type in self.cfg.graph.structure[0]:
                    log.info(f'Adding edges of type {edge_type} ...')
                    assert edge_type[0] in teamsvecs.keys() and teamsvecs[edge_type[0]] is not None
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

    def train(self, teamsvecs, indexes):
        self._prep(teamsvecs, None)
        self.cfg.model = self.cfg[self.name] #gnn.n2v or gnn.gs --> gnn.model
        prefix = self.output + f'/d{self.cfg.model.d}.e{self.cfg.model.e}.ns{self.cfg.model.ns}.{self.name}'
        postfix = f'{".pre" if self.cfg.graph.pre else ""}.{self.cfg.graph.dup_edge}.{self.cfg.graph.structure[1]}'

        # replace the 1 dimensional node features with pretrained d2v skill vectors of required dimension
        if self.cfg.graph.pre: self.d2v_node_features(indexes, teamsvecs)

        log.info(f'Training {self.name} ... ')
        loader = None; optimizer = None
        if self.name == 'n2v':
            output = f'.w{self.cfg.model.w}.wl{self.cfg.model.wl}.wn{self.cfg.model.wn}'
            from torch_geometric.nn import Node2Vec
            # ImportError: 'Node2Vec' requires either the 'pyg-lib' or 'torch-cluster' package
            install_import(f'torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-{Gnn.torch.__version__}.html', 'torch_cluster')
            self.model = Node2Vec((data:=(self.data.to_homogeneous() if isinstance(self.data, Gnn.pyg.data.HeteroData) else self.data)).edge_index,
                                 embedding_dim=self.cfg.model.d,
                                 walk_length=self.cfg.model.wl,
                                 context_size=self.cfg.model.w,
                                 walks_per_node=self.cfg.model.wn,
                                 num_negative_samples=self.cfg.model.ns).to(self.device)
            loader = self.model.loader(batch_size=self.cfg.model.b, shuffle=True, num_workers=0)#os.cpu_count()) not working in windows!
            optimizer = Gnn.torch.optim.Adam(list(self.model.parameters()), lr=self.cfg.model.lr)

            self.model.train()
            for epoch in range(1, self.cfg.model.e + 1):
                b_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                    loss.backward(); optimizer.step(); b_loss += loss.item()
                b_loss /= len(loader)
                log.info(f'Epoch {epoch}, Loss: {b_loss:.4f}')

            self.model.eval()
            Gnn.torch.save({
                'model_state_dict': self.model.state_dict(),
                'edge_index': data.edge_index,  # to re-init the model
                'params': {'d': self.model.embedding_dim,'wl': self.model.walk_length,'w': self.model.context_size,'wn': self.model.walks_per_node,'ns': self.model.num_negative_samples,'sparse': True},
                'node_type': getattr(data, 'node_type', None),  # required to know the node_type after to_homo()
                'node_type_names': getattr(self.data, 'node_types', None) #the HeteroData has node_types but not the Data!
            }, prefix + output + postfix)

            embeddings = self.model.embedding.weight.data.cpu()
            if isinstance(self.data, Gnn.pyg.data.HeteroData):
                node_type_names = self.data.node_types  # ['a', 'b', 'c']
                node_type_tensor = data.node_type  # tensor of shape [num_nodes]
                for i, type_name in enumerate(node_type_names):
                    mask = (node_type_tensor == i)
                    type_embeddings = embeddings[mask]  # shape: [num_nodes_of_type, 128]
                    print(f"Node type: {type_name}, Shape: {type_embeddings.shape}")
            return

        # elif self.name in {'gs', 'gin', 'gat', 'gatv2', 'han', 'gine', 'lant'}:
        #     self.init_model(emb_output)
        #     self.train(self.self.cfg.model.e)
        #     return
        # # gcn (for homogeneous only)
        # # TODO: make hetreo of any type to homo
        # elif self.name == 'gcn':
        #     from gcn_old import Gcn as GCNModel
        #     self.model = GCNModel(hidden_channels=10, data=t2v.data)
        # elif self.model == 'm2v':
        #     from m2v import M2V
        #     from torch_geometric.nn import MetaPath2Vec
        #     t2v = M2V(teamsvecs, indexes, params.settings, output_, emb_output)
        #     t2v.name = 'm2v'
        #     t2v.init()  # call the m2v's init
        #     t2v.model = MetaPath2Vec(t2v.data.edge_index_dict, embedding_dim=t2v.settings['d'],
        #                              metapath=t2v.settings['metapath'][edge_type[1]],
        #                              walk_length=t2v.settings['walk_length'],
        #                              context_size=t2v.settings['context_size'],
        #                              walks_per_node=t2v.settings['walks_per_node'],
        #                              num_negative_samples=t2v.settings['ns'],
        #                              sparse=True).to(t2v.device)
        #     t2v.init_model()
        #     t2v.train(t2v.settings['e'])
        #     t2v.model.eval()
        #     emb = {}
        #     node_types = t2v.data._node_store_dict.keys()
        #     for node_type in node_types: emb[node_type] = t2v.model(node_type)  # output of embeddings
        #     embedding_output = f'{t2v.emb_output}.emb.pt'
        #     Gnn.torch.save(emb, embedding_output, pickle_protocol=4)
        #     return
        #
        # self.optimizer = Gnn.torch.optim.Adam(list(self.model.parameters()), lr=self.cfg.model.lr)
        # self.train(self.cfg.model.e, self.cfg.save_per_epoch)
        # if self.name == 'lant': self.model.learn(self, self.cfg.model.e)  # built-in validation inside lant_encoder class
        #
        # self.plot_points()
        # log.info(self)

    def d2v_node_features(self, indexes, teamsvecs):
        log.info(f'Loading pretrained d2v embeddings {self.cfg.graph.pre} to initialize node features, or if not exist, train d2v embeddings from scratch ...')
        from .d2v import D2v
        d2v_filename = os.path.basename(self.cfg.graph.pre)
        d2v_cfg = str2cfg(d2v_filename)
        d2v_cfg.embtype = d2v_filename.split('.')[-1] # Check emb.d2v.D2v.train()
        d2v_cfg.lr = self.cfg.model.lr
        # simple lazy load, or train from scratch if the file not found!
        d2v_obj = D2v(os.path.dirname(self.cfg.graph.pre), self.device, d2v_cfg).train(teamsvecs, indexes)
        # the order is NOT correct in d2v, i.e., vecs[0] may be for vecs['s20']. Call D2v.natsortvecs(d2v_obj.model.wv)
        # d2v = Doc2Vec.load(self.cfg.graph.pre)
        for node_type in self.data.node_types:
            if node_type == 'team':
                # no issue as the docs are 0-based indexed 0 --> '0'
                assert d2v_obj.model.docvecs.vectors.shape[0] == teamsvecs['skill'].shape[0]  # correct number of embeddings per team
                indices = [d2v_obj.model.docvecs.key_to_index[str(i)] for i in range(len(d2v_obj.model.docvecs))]
                assert np.allclose(d2v_obj.model.docvecs.vectors, d2v_obj.model.docvecs.vectors[indices])
                # assert np.array_equal(d2v_obj.model.docvecs.vectors, d2v_obj.model.docvecs.vectors[indices])
                # (d2v_obj.model.docvecs.vectors[2] == d2v_obj.model.docvecs['2']).all()
                self.data[node_type].x = Gnn.torch.tensor(d2v_obj.model.docvecs.vectors)  # team vectors (dv) for 'team' nodes, else individual node vectors (wv)
            else:  # either 'skill' or 'member'
                if d2v_obj.model.wv.vectors.shape[0] == teamsvecs[node_type].shape[1]:  # correct number of embeddings per skills xor members
                    self.data[node_type].x = Gnn.torch.tensor(D2v.natsortvecs(d2v_obj.model.wv))
        if d2v_obj.model.wv.vectors.shape[0] == teamsvecs['skill'].shape[1] + teamsvecs['member'].shape[1]:
            ordered_vecs = Gnn.torch.tensor(D2v.natsortvecs(d2v_obj.model.wv))
            if 'member' in self.data.node_types: self.data['member'].x = ordered_vecs[:teamsvecs['member'].shape[1]]  # the first part is all m*
            if 'skill' in self.data.node_types: self.data['skill'].x = ordered_vecs[teamsvecs['member'].shape[1]:]  # the remaining is s*

    # # settings = the settings for this particular gnn model
    # # emb_output = the path for the embedding output and model output storage
    # def init_model(self):
    #
    #     #if self.model_name == 'han': self.metapaths = self.settings['metapaths'][self.graph_type]
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
    #     Gnn.torch.cuda.empty_cache()
    #     train_data.to(self.device)
    #     # the train_data is needed to collect info about the metadata
    #     from encoder import Encoder
    #     self.model = Encoder(hidden_channels=self.d, data=train_data, model_name=self.model).to(self.device)
    #     # if self.model_name == 'lant':
    #     #     from lant_encoder import Encoder
    #     #     model = Encoder(hidden_channels=self.d, data=train_data)
    #     self.optimizer = Gnn.torch.optim.Adam(self.model.parameters(), lr=self.cfg.model.lr)

    # def save_emb(self):
    #     with Gnn.torch.no_grad():
    #         for node_type in self.data.node_types: self.data[node_type].n_id = Gnn.torch.arange(self.data[node_type].x.shape[0])
    #         self.data.to(self.device)
    #         # for simplicity, we just pass seed_edge_type = edge_types[0]. This does not impact any output
    #         emb = self.model(self.data, self.edge_types[0], self.is_directed, emb=True)
    #         embedding_output = f'{self.output}/{self.cfg.graph.structure[1]}.{self.cfg.graph.dup_edge if self.cfg.graph.dup_edge else "dup"}.{cfg2str(self.cfg)}.emb.{self.model}'
    #         Gnn.torch.save(emb, embedding_output, pickle_protocol=4)
    #         log.info(f'Saved embedding as {embedding_output}')
    #     # eval_batch(test_loader, is_directed)
    #     Gnn.torch.cuda.empty_cache()
    #     # torch.save(self.model.state_dict(), f'{self.model_output}/gnn_model.pt', pickle_protocol=4)
    #     #to load later by: self.model.load_state_dict(torch.load(f'{self.output}/gnn_model.pt'))

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
    #         Gnn.torch.cuda.empty_cache()
    #         # train for loaders of all edge_types, e.g : train_loader['skill','to','team'], train_loader['member','to','team']
    #         for seed_edge_type in self.cfg.graph.supervision_edge_types:
    #             print(f'epoch {epoch:03d} : batching for train_loader for seed_edge_type : {seed_edge_type}')
    #             for sampled_data in loader[seed_edge_type]:
    #                 self.optimizer.zero_grad()
    #                 sampled_data.to(self.device)
    #                 pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #                 # The ground_truth and the pred shapes should be 1-dimensional
    #                 # we squeeze them after generation
    #                 if (type(sampled_data) == Gnn.pyg.data.HeteroData): ground_truth = sampled_data[seed_edge_type].edge_label
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
    #     self.plot_graph(Gnn.torch.arange(1, epochs_taken + 1, 1), loss_array, val_loss_array, fig_output=fig_output)
    #     fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.val_auc_per_epoch.png'
    #     self.plot_graph(Gnn.torch.arange(1, epochs_taken + 1, 1), val_auc_array, xlabel='Epochs', ylabel='Val AUC', title=f'Validation AUC vs Epochs for Embedding Generation', fig_output=fig_output)
    #
    # def split(self, data, manual_neg_samples):
    #     edge_types = None; rev_edge_types = None
    #     if (type(data) == Gnn.pyg.data.HeteroData):#from torch_geometric.data import HeteroData
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
    #         new_edge_label_index = Gnn.torch.cat([pos_edge_index, neg_edge_index], dim=1)
    #         new_edge_label = Gnn.torch.cat([Gnn.torch.ones(num_pos_samples), Gnn.torch.zeros(num_neg_samples)], dim=0)
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
    #     with Gnn.torch.no_grad():
    #         for seed_edge_type in self.cfg.graph.supervision_edge_types:
    #             for sampled_data in loader[seed_edge_type]:
    #                 sampled_data.to(self.device)
    #                 tmp_pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #                 if (type(sampled_data) == Gnn.pyg.data.HeteroData): # we have ground_truths per edge_label_index
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
    #     pred = Gnn.torch.cat(preds, dim=0).cpu().numpy()
    #     ground_truth = Gnn.torch.cat(ground_truths, dim=0).cpu().numpy()
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
