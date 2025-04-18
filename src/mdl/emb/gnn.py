import os, itertools, pickle, logging
from tqdm import tqdm

log = logging.getLogger(__name__)

from pkgmgr import *
from .team2vec import Team2Vec

class Gnn(Team2Vec):

    def __init__(self, dim, output, device, cgf):
        super().__init__(dim, output, device, cgf)
        self.loader = None
        self.optimizer = None

        Gnn.torch = install_import(cgf.pytorch, 'torch')
        Gnn.pyg = install_import('torch_geometric==2.6.1', 'torch_geometric')

        # F = install_import('', 'torch.nn.functional')
        # LinkNeighborLoader = install_import('', 'torch_geometric.laoder', 'LinkNeighborLoader')
        # from mdl.earlystopping import EarlyStopping
        # from lant_encoder import Encoder as LANT_Encoder

    def _prep(self, teamsvecs, indexes): # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_self_loops
        structure = eval(self.cfg.graph.structure)
        file = self.output + f'/{structure[1]}graph.pkl'

        if not isinstance(structure[0], list):#homo
            log.info(f'Creating a homo graph with {structure[0]} node type ...')
            teams = teamsvecs[structure[0]] #TODO: if node_type == 'team'
            edges = []
            for i, row in enumerate(tqdm(teams, total=teams.shape[0])):
                for t in itertools.combinations(row.nonzero()[1], 2): edges += [t]
            edge_index = Gnn.torch.tensor(edges, dtype=Gnn.torch.long).t().contiguous() #[1,2][1,3][2,4] >> [1,1,2][2,3,4]
            nodes = Gnn.torch.tensor([[0]] * teams.shape[1], dtype=Gnn.torch.float)
            self.data = Gnn.pyg.data.Data(x=nodes, edge_index=edge_index, edge_attr=Gnn.torch.tensor([1] * len(edges), dtype=Gnn.torch.long))
        else:
            log.info(f'Creating a hetero graph of type {structure[0]} ...')
            self.data = Gnn.pyg.data.HeteroData()
            node_types = set()
            #edges
            for edge_type in structure[0]:
                log.info(f'Adding edges of type {edge_type} ...')
                teams = teamsvecs[edge_type[0]] # take one part of an edge from here
                edges = []
                for i, row1 in enumerate(tqdm(teams, total=teams.shape[0])):
                    row2 = teamsvecs[edge_type[2]][i] if edge_type[2] != 'team' else [i] # take the other part of the edge from here
                    if edge_type[0] == edge_type[2]:
                        for t in itertools.combinations(row1.nonzero()[1], 2): edges += [t] # now add edges from all members of part 1 to part 2 (in this case, both are the same, so we take combinations of 2)
                    else:
                        for t in itertools.product(row1.nonzero()[1], row2.nonzero()[1] if edge_type[2] != 'team' else row2): edges += [t] # now add edges from all members of part 1 to part 2

                '''
                For edge_type ('skill', 'to', 'skill') and edges = [(0, 0), (0, 1), (1, 0), (1, 1)] from the previous step for one single team, we are looking at two identical edges
                s-s (0,1)
                s-s (1,0)
                so we need to only consider combinations for the s-s or m-m edges. In this case, [(0, 0), (0, 1), (1, 0), (1, 1)] >> [(0, 0), (0, 1), (1, 1)] for one single team
                '''

                self.data[edge_type].edge_index = Gnn.torch.tensor(edges, dtype=Gnn.torch.long).t().contiguous()
                self.data[edge_type].edge_attr = Gnn.torch.tensor([1] * len(edges), dtype=Gnn.torch.long)
                node_types = node_types.union({edge_type[0], edge_type[2]})
            #nodes
            for node_type in node_types: self.data[node_type].x = Gnn.torch.tensor([[0]] * (teamsvecs[node_type].shape[1] if node_type != 'team' else teamsvecs['skill'].shape[0]), dtype=Gnn.torch.float)

        # if not self.settings['dir']:
        log.info('To undirected graph ...')
        transform = Gnn.pyg.transforms.ToUndirected(reduce=self.cfg.graph.dup_edge) # this will also aggregate the edge features

        # # we will only create reverse edges if the graph is undirected
        # # create reverse edges for s-s and m-m edge_types
        # for edge_type in self.data.edge_types:
        #     # add reverse edge_types for s-s and e-e edge_types
        #     if edge_type[0] == edge_type[2]:
        #         rev_edge_type = (edge_type[0], f'rev_{edge_type[1]}', edge_type[2])  # reverse the relation
        #         log.info(f'Creating {rev_edge_type} manually')
        #         self.data[rev_edge_type].edge_index = self.data[edge_type].edge_index  # basically the same edge_index
        #         self.data[rev_edge_type].edge_attr = self.data[edge_type].edge_attr  # basically the same edge_attr
        # # apply the final transform
        self.data = transform(self.data)

        if self.cfg.graph.dup_edge:
            log.info(f'To merge duplicate edges by {self.cfg.graph.dup_edge} weights/features ...')
            transform = Gnn.pyg.transforms.RemoveDuplicatedEdges(key=['edge_attr'], reduce=self.cfg.graph.dup_edge)
            self.data = transform(self.data)

        self.data.validate(raise_on_error=True)
        with open(file, 'wb') as f: pickle.dump(self.data, f)
        return self.data

    def train(self, epochs, teamsvecs, indexes):
        self._prep(teamsvecs, indexes)
    # # settings = the settings for this particular gnn model
    # # emb_output = the path for the embedding output and model output storage
    # def init_model(self, emb_output):
    #
    #     # getting from the gnn params
    #     self.e = self.settings['e']
    #     self.d = self.settings['d']
    #     self.b = self.settings['b']
    #     self.ns = self.settings['ns']
    #     self.nn = self.settings['nn']
    #     self.agg = self.settings['agg']
    #     self.dir = self.settings['dir']
    #     self.graph_type = self.settings['graph_types']
    #     if self.model_name == 'han': self.metapaths = self.settings['metapaths'][self.graph_type]
    #
    #     # e.g : domain = 'imdb/title.basics.tsv.filtered.mt5.ts2'
    #     # self.filepath = f'../../data/preprocessed/{domain}/gnn/{graph_type}.{dir}.{agg}.data.pkl'
    #     self.model_output = emb_output
    #     if not os.path.isdir(self.model_output): os.makedirs(self.model_output)
    #     self.is_directed = self.data.is_directed()
    #
    #     # initialize the model based on the param for training
    #     train_data, val_data, test_data, self.edge_types, self.rev_edge_types = self.define_splits(self.data)
    #
    #     # create separate loaders for separate seed edge_types
    #     self.train_loader, self.val_loader, self.test_loader = {}, {}, {}
    #
    #     # these settings are central settings set in main
    #     for edge_type in params.settings['graph']['supervision_edge_types']:
    #         self.train_loader[edge_type] = self.create_mini_batch_loader(train_data, edge_type, 'train')
    #         self.val_loader[edge_type] = self.create_mini_batch_loader(val_data, edge_type, 'val')
    #         # self.test_loader[edge_type] = self.create_mini_batch_loader(test_data, edge_type, 'test') # we dont need a test loader as of now
    #
    #     print(f"Device: '{self.device}'")
    #     torch.cuda.empty_cache()
    #     train_data.to(self.device)
    #     # the train_data is needed to collect info about the metadata
    #     self.model, self.optimizer = self.create_gnn_model(train_data)
    #
    # def train(self, epochs, save_per_epoch=False):
    #     if self.model_name == 'lant':
    #         self.model.learn(self, epochs) # built-in validation inside lant_encoder class
    #     else:
    #         self.learn(self.train_loader, epochs)
    #         self.eval(self.val_loader)
    #
    #     print(f'-------------- ending eval --------------')
    #
    #     # store the embeddings
    #     with torch.no_grad():
    #         for node_type in self.data.node_types:
    #             self.data[node_type].n_id = torch.arange(self.data[node_type].x.shape[0])
    #         self.data.to(self.device)
    #
    #         # for simplicity, we just pass seed_edge_type = edge_types[0]. This does not impact any output
    #         emb = self.model(self.data, self.edge_types[0], self.is_directed, emb=True)
    #         embedding_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.{"dir" if self.dir else "undir"}.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.emb.pt'
    #         torch.save(emb, embedding_output, pickle_protocol=4)
    #         print(f'\nsaved embedding as : {embedding_output} ..............\n')
    #     # eval_batch(test_loader, is_directed)
    #     torch.cuda.empty_cache()
    #     # torch.save(self.model.state_dict(), f'{self.model_output}/gnn_model.pt', pickle_protocol=4)
    #     #to load later by: self.model.load_state_dict(torch.load(f'{self.output}/gnn_model.pt'))
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
    #
    # def define_splits(self, data):
    #
    #     if (type(data) == HeteroData):
    #         num_edge_types = len(data.edge_types)
    #
    #         # directed graph means we dont have any reverse edges
    #         if (data.is_directed()):
    #             edge_types = data.edge_types
    #             rev_edge_types = None
    #         else:
    #             edge_types = data.edge_types[:num_edge_types // 2]
    #             rev_edge_types = data.edge_types[num_edge_types // 2:]
    #     else:
    #         edge_types = None
    #         rev_edge_types = None
    #
    #     transform = T.RandomLinkSplit(
    #         num_val=0.1,
    #         num_test=0.0,
    #         disjoint_train_ratio=0.3,
    #         neg_sampling_ratio=0.0,             # we leave negative sampling to the mini_batch_loaders
    #         add_negative_train_samples=False,
    #         edge_types=edge_types,
    #         rev_edge_types=rev_edge_types,
    #     )
    #
    #     train_data, val_data, test_data = transform(data)
    #
    #     train_data.validate(raise_on_error=True)
    #     val_data.validate(raise_on_error=True)
    #     test_data.validate(raise_on_error=True)
    #
    #     # if han, add metapaths to the metadata
    #     if self.model_name == 'han':
    #         from torch_geometric.transforms import AddMetaPaths
    #         train_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False,
    #                                   drop_unconnected_node_types=False)(train_data)
    #         val_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False,
    #                                 drop_unconnected_node_types=False)(val_data)
    #
    #     return train_data, val_data, test_data, edge_types, rev_edge_types
    #
    # # separately handle the negative sampling part
    # def define_splits_and_neg_sampling(self, data):
    #
    #     if (type(data) == HeteroData):
    #         num_edge_types = len(data.edge_types)
    #
    #         # directed graph means we dont have any reverse edges
    #         if (data.is_directed()):
    #             edge_types = data.edge_types
    #             rev_edge_types = None
    #         else:
    #             edge_types = data.edge_types[:num_edge_types // 2]
    #             rev_edge_types = data.edge_types[num_edge_types // 2:]
    #     else:
    #         edge_types = None
    #         rev_edge_types = None
    #
    #     transform = T.RandomLinkSplit(
    #         num_val=0.1,
    #         num_test=0.0,
    #         disjoint_train_ratio=0.3,
    #         neg_sampling_ratio=0.0,             # leaving this for manual neg_sampling
    #         add_negative_train_samples=False,
    #         edge_types=edge_types,
    #         rev_edge_types=rev_edge_types,
    #     )
    #
    #     train_data, val_data, test_data = transform(data)
    #
    #     # manual negative sampling
    #     for data_type in [train_data, val_data]: self.manual_negative_sampling(data_type, edge_types, self.ns)
    #
    #     train_data.validate(raise_on_error=True)
    #     val_data.validate(raise_on_error=True)
    #     test_data.validate(raise_on_error=True)
    #
    #     # if han, add metapaths to the metadata
    #     if self.model_name == 'han':
    #         from torch_geometric.transforms import AddMetaPaths
    #         train_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False,
    #                                   drop_unconnected_node_types=False)(train_data)
    #         val_data = AddMetaPaths(metapaths=self.metapaths, drop_orig_edge_types=False,
    #                                 drop_unconnected_node_types=False)(val_data)
    #
    #     return train_data, val_data, test_data, edge_types, rev_edge_types
    #
    # # Manual negative sampling
    # def manual_negative_sampling(self, data_type, edge_types, num_samples_ratio):
    #     # handle neg_sampling manually
    #     from torch_geometric.utils import negative_sampling
    #
    #     for edge_type in edge_types:
    #         pos_edge_index = data_type[edge_type].edge_label_index
    #         num_pos_samples = pos_edge_index.shape[1]
    #         num_neg_samples = int(num_samples_ratio * num_pos_samples)
    #
    #         neg_edge_index = negative_sampling(
    #             edge_index=data_type[edge_type].edge_index,
    #             num_nodes=(data_type[edge_type[0]].num_nodes,data_type[edge_type[2]].num_nodes),  # Specify number of nodes if necessary
    #             num_neg_samples=num_neg_samples,
    #             method='sparse'
    #         )
    #
    #         # Concatenate positive and negative edges
    #         new_edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    #         new_edge_label = torch.cat([torch.ones(num_pos_samples), torch.zeros(num_neg_samples)], dim=0)
    #
    #         data_type[edge_type].edge_label_index = new_edge_label_index
    #         data_type[edge_type].edge_label = new_edge_label
    #
    #     return data_type
    #
    # # d = dim
    # # we pass only train_data for providing metadata to the model
    # def create_gnn_model(self, data):
    #
    #     from encoder import Encoder as Encoder
    #     if self.model_name == 'lant':
    #         model = LANT_Encoder(hidden_channels=self.d, data=data)
    #     else:
    #         model = Encoder(hidden_channels=self.d, data=data, model_name= self.model_name)
    #
    #     print(model)
    #     print(f'\nDevice = {self.device}')
    #
    #     model = model.to(self.device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     return model, optimizer
    #
    # # create a mbatch for a given split_data (mode = train / val / test)
    # def create_mini_batch_loader(self, split_data, seed_edge_type, mode):
    #     # Define seed edges:
    #     # we pick only a single edge_type to feed edge_label_index (need to verify this approach)
    #
    #     # neg_sampling in val or test loader causes doubling the edge_label weight producing 2.0 instead of values 1.0 (need to test)
    #     neg_sampling_ratio = self.ns                                  # updated after v5
    #     batch_size = self.b if mode == 'train' else (3 * self.b)
    #     shuffle = True
    #
    #     print(f'mini batch loader for mode {mode}')
    #     mini_batch_loader = LinkNeighborLoader(
    #         data=split_data,
    #         num_neighbors=self.nn,
    #         neg_sampling='binary',
    #         neg_sampling_ratio=neg_sampling_ratio,             # prev : neg_sampling = None
    #         edge_label_index=(seed_edge_type, split_data[seed_edge_type].edge_label_index),
    #         edge_label=split_data[seed_edge_type].edge_label,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #     )
    #     return mini_batch_loader
    #
    # def learn(self, loader, epochs):
    #     start = time.time()
    #     epochs_taken = 0
    #     loss_array = []
    #     val_auc_array = []
    #     val_loss_array = []
    #     earlystopping = EarlyStopping(patience=5, verbose=True, delta=0.001,
    #                                   path=f"{self.model_output}/state_dict_model.e{epochs}.pt", trace_func=print, save_model=False)
    #     for epoch in range(1, epochs + 1):
    #         self.optimizer.zero_grad()  # ensuring clearing out the gradients before each validation loop
    #
    #         total_loss = 0
    #         total_examples = 0
    #         torch.cuda.empty_cache()
    #         # train for loaders of all edge_types, e.g : train_loader['skill','to','team'], train_loader['member','to','team']
    #         for seed_edge_type in params.settings['graph']['supervision_edge_types']:
    #             print(f'epoch {epoch:03d} : batching for train_loader for seed_edge_type : {seed_edge_type}')
    #             for sampled_data in loader[seed_edge_type]:
    #                 self.optimizer.zero_grad()
    #
    #                 sampled_data.to(self.device)
    #                 pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #                 # The ground_truth and the pred shapes should be 1-dimensional
    #                 # we squeeze them after generation
    #                 if (type(sampled_data) == HeteroData):
    #                     ground_truth = sampled_data[seed_edge_type].edge_label
    #                 else:
    #                     ground_truth = sampled_data.edge_label
    #                 loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #                 loss.backward()
    #                 self.optimizer.step()
    #
    #                 total_loss += float(
    #                     loss) * pred.numel()  # each batch might not contain same number of predictions, so we normalize it using the individual number of preds each turn
    #                 total_examples += pred.numel()
    #         avg_loss = (total_loss / total_examples)
    #         if (epoch % 10 == 0):
    #             print(f"\n.............Epoch: {epoch:03d}, Loss: {avg_loss:.6f}.............\n")
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
    #     self.plot_graph(torch.arange(1, epochs_taken + 1, 1), loss_array, val_loss_array, fig_output=fig_output)
    #     fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.val_auc_per_epoch.png'
    #     self.plot_graph(torch.arange(1, epochs_taken + 1, 1), val_auc_array, xlabel='Epochs', ylabel='Val AUC',
    #                title=f'Validation AUC vs Epochs for Embedding Generation', fig_output=fig_output)
    #     print(f'\nit took {(time.time() - start) / 60} mins || {(time.time() - start) / 3600} hours to train the model\n')
    #
    # @torch.no_grad
    # def eval(self, loader):
    #     from sklearn.metrics import roc_auc_score
    #     preds = []
    #     ground_truths = []
    #     total_loss = 0
    #     total_examples = 0
    #     for seed_edge_type in params.settings['graph']['supervision_edge_types']:
    #         for sampled_data in loader[seed_edge_type]:
    #             sampled_data.to(self.device)
    #             tmp_pred = self.model(sampled_data, seed_edge_type, self.is_directed)
    #             if (type(sampled_data) == HeteroData):
    #                 # we have ground_truths per edge_label_index
    #                 tmp_ground_truth = sampled_data[seed_edge_type].edge_label
    #             else:
    #                 tmp_ground_truth = sampled_data.edge_label
    #
    #             assert tmp_pred.shape == tmp_ground_truth.shape
    #             loss = F.binary_cross_entropy_with_logits(tmp_pred, tmp_ground_truth)
    #             total_loss += float(loss) * tmp_pred.numel()
    #             total_examples += tmp_pred.numel()
    #
    #             preds.append(tmp_pred)
    #             ground_truths.append(tmp_ground_truth)
    #
    #     pred = torch.cat(preds, dim=0).cpu().numpy()
    #     ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    #     loss = total_loss / total_examples
    #     auc = roc_auc_score(ground_truth, pred)
    #     print()
    #     print(f'Val loss : {loss:.6f}')
    #     print(f"Val AUC: {auc:.6f}\n")
    #     # print(f'................... ending eval...................\n')
    #     return loss, auc
    #
