import numpy as np, math, os, itertools, pickle, time, json
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from mdl.earlystopping import EarlyStopping
from team2vec import Team2Vec

class Gnn(Team2Vec):

    def __init__(self, teamsvecs, indexes, settings, output):
        super().__init__(teamsvecs, indexes, settings, output)

        self.loader = None
        self.optimizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # this line enables to produce separate directories for separate graph_types
        # if not os.path.isdir(self.output): os.makedirs(self.output)

    def create(self, file):
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_self_loops

        if not isinstance(self.settings['edge_types'][0], list):#homo
            print(f'Creating a homo graph with {self.settings["edge_types"][0]} node type ...')
            teams = self.teamsvecs[self.settings['edge_types'][0]] #TODO: if node_type == 'team'
            edges = []
            for i, row in enumerate(tqdm(teams, total=teams.shape[0])):
                for t in itertools.combinations(row.nonzero()[1], 2): edges += [t]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() #[1,2][1,2] >> [1,1][2,2]
            nodes = torch.tensor([[0]] * teams.shape[1], dtype=torch.float)
            self.data = Data(x=nodes, edge_index=edge_index, edge_attr=torch.tensor([1] * len(edges), dtype=torch.long))
        else:
            print(f'Creating a hetero graph ...')
            self.data = HeteroData()
            node_types = set()
            #edges
            for edge_type in self.settings['edge_types'][0]:
                print(f'Adding edges of type {edge_type} ...')
                teams = self.teamsvecs[edge_type[0]]
                edges = []
                for i, row1 in enumerate(tqdm(teams, total=teams.shape[0])):
                    row2 = self.teamsvecs[edge_type[2]][i] if edge_type[2] != 'team' else [i]
                    for t in itertools.product(row1.nonzero()[1], row2.nonzero()[1] if edge_type[2] != 'team' else row2): edges += [t]
                self.data[edge_type].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                self.data[edge_type].edge_attr = torch.tensor([1] * len(edges), dtype=torch.long)
                node_types = node_types.union({edge_type[0], edge_type[2]})
            #nodes
            for node_type in node_types: self.data[node_type].x = torch.tensor([[0]] * (self.teamsvecs[node_type].shape[1] if node_type != 'team' else self.teamsvecs['id'].shape[0]), dtype=torch.float)

        if not self.settings['dir']:
            print('To undirected graph ...')
            import torch_geometric.transforms as T
            transform = T.ToUndirected()
            self.data = transform(self.data)

        if self.settings['dup_edge']:
            print(f'To reduce duplicate edges by {self.settings["dup_edge"]} ...')
            import torch_geometric.transforms as T
            transform = T.RemoveDuplicatedEdges(key=["edge_attr"], reduce=self.settings['dup_edge'])
            self.data = transform(self.data)

        self.data.validate(raise_on_error=True)
        with open(file, 'wb') as f: pickle.dump(self.data, f)
        return self.data

    # settings = the settings for this particular gnn model
    # emb_output = the path for the embedding output and model output storage
    def init_model(self, emb_output):

        # getting from the gnn params
        self.e = self.settings['e']
        self.d = self.settings['d']
        self.b = self.settings['b']
        self.ns = self.settings['ns']
        self.nn = self.settings['nn']
        self.agg = self.settings['agg']
        self.graph_type = self.settings['graph_type']

        # e.g : domain = 'imdb/title.basics.tsv.filtered.mt5.ts2'
        # self.filepath = f'../../data/preprocessed/{domain}/gnn/{graph_type}.undir.{agg}.data.pkl'
        self.model_output = emb_output
        if not os.path.isdir(self.model_output): os.makedirs(self.model_output)
        self.is_directed = self.data.is_directed()

        # initialize the model based on the param for training
        train_data, val_data, test_data, self.edge_types, self.rev_edge_types = self.define_splits(self.data)

        # create separate loaders for separate seed edge_types
        self.train_loader, self.val_loader, self.test_loader = {}, {}, {}
        for edge_type in self.edge_types:
            self.train_loader[edge_type] = self.create_mini_batch_loader(train_data, edge_type, 'train')
            self.val_loader[edge_type] = self.create_mini_batch_loader(val_data, edge_type, 'val')
            self.test_loader[edge_type] = self.create_mini_batch_loader(test_data, edge_type, 'test')

        print(f"Device: '{self.device}'")
        torch.cuda.empty_cache()
        train_data.to(self.device)
        # the train_data is needed to collect info about the metadata
        self.model, self.optimizer = self.create_gnn_model(train_data)

    def train(self, epochs, save_per_epoch=False):
        self.learn(self.train_loader, epochs)
        self.eval(self.val_loader)
        print(f'-------------- ending eval --------------')
        # store the embeddings
        with torch.no_grad():
            for node_type in self.data.node_types:
                self.data[node_type].n_id = torch.arange(self.data[node_type].x.shape[0])
            self.data.to(self.device)
            # for simplicity, we just pass seed_edge_type = edge_types[0]. This does not impact any output
            emb = self.model(self.data, self.edge_types[0], self.is_directed, emb=True)
            embedding_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.emb.pt'
            torch.save(emb, embedding_output, pickle_protocol=4)
            print(f'\nsaved embedding as : {embedding_output} ..............\n')
        # eval_batch(test_loader, is_directed)
        torch.cuda.empty_cache()
        # torch.save(self.model.state_dict(), f'{self.model_output}/gnn_model.pt', pickle_protocol=4)
        #to load later by: self.model.load_state_dict(torch.load(f'{self.output}/gnn_model.pt'))


        # @Hossein Fani
        # print(f'It took {time.time() - t_start_time} to train the model.')
        # with open(f'{self.model_output}/train_valid_loss.json', 'w') as outfile: json.dump({'train': train_loss_values, 'valid': valid_loss_values}, outfile)
        # plt.figure()
        # plt.plot(train_loss_values, label='Training Loss')
        # plt.plot(valid_loss_values, label='Validation Loss')
        # plt.legend(loc='upper right')
        # plt.title(f'Training and Validation Losses per Epoch')
        # plt.savefig(f'{self.model_output}/train_valid_loss.png', dpi=100, bbox_inches='tight')
        # plt.show()

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        z = self.model()
        #https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html
        #TODO: the default test is LR on node labels, but we should bring it to team formation test
        #acc = model.test(train_z=z[self.data.train_mask], train_y=data.y[self.data.train_mask], test_z=z[self.data.test_mask], test_y=data.y[self.data.test_mask], max_iter=150)
        return 0# acc

    @torch.no_grad()
    def plot_points(self):
        from sklearn.manifold import TSNE
        self.model.eval()
        z = self.model().cpu().numpy()
        z = TSNE(n_components=2).fit_transform(z)
        #y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        # for i in range(dataset.num_classes):
        #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.scatter(z[:, 0], z[:, 1], s=20)
        plt.axis('off')
        plt.savefig(f'{self.output}/{self.model_name}/tsne.png', dpi=100, bbox_inches='tight')
        plt.show()

    # made specifically for the gnn training
    def plot_graph(self, x, y, *args, xlabel='Epochs', ylabel='Loss', title='Loss vs Epochs', fig_output='plot.png'):
        plt.plot(x, y, label='Train')  # Plot the first set of data

        if len(args) > 0:
            plt.plot(x, args[0], label='Valid')  # Plot the second set of data
            plt.legend()  # Add legend if there are two sets of data

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        print(f'\nsaving figure as : {fig_output}\n')
        plt.savefig(fig_output)
        plt.clf()
        # plot.show()

    def define_splits(self, data):

        if (type(data) == HeteroData):
            num_edge_types = len(data.edge_types)

            # directed graph means we dont have any reverse edges
            if (data.is_directed()):
                edge_types = data.edge_types
                rev_edge_types = None
            else:
                edge_types = data.edge_types[:num_edge_types // 2]
                rev_edge_types = data.edge_types[num_edge_types // 2:]
        else:
            edge_types = None
            rev_edge_types = None

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.0,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=self.ns,
            add_negative_train_samples=True,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
        )

        train_data, val_data, test_data = transform(data)

        train_data.validate(raise_on_error=True)
        val_data.validate(raise_on_error=True)
        test_data.validate(raise_on_error=True)

        return train_data, val_data, test_data, edge_types, rev_edge_types

    # d = dim
    # we pass only train_data for providing metadata to the model
    def create_gnn_model(self, data):

        from encoder import Encoder as Encoder
        model = Encoder(hidden_channels=self.d, data=data, model_name= self.model_name)

        print(model)
        print(f'\nDevice = {self.device}')

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer

    # create a mbatch for a given split_data (mode = train / val / test)
    def create_mini_batch_loader(self, split_data, seed_edge_type, mode):
        # Define seed edges:
        # we pick only a single edge_type to feed edge_label_index (need to verify this approach)

        # neg_sampling in val or test loader causes doubling the edge_label weight producing 2.0 instead of values 1.0 (need to test)
        neg_sampling = self.ns if mode == 'train' else None
        batch_size = self.b if mode == 'train' else (3 * self.b)
        shuffle = True if mode == 'train' else False

        print(f'mini batch loader for mode {mode}')
        mini_batch_loader = LinkNeighborLoader(
            data=split_data,
            num_neighbors=self.nn,
            neg_sampling_ratio=neg_sampling,  # prev : neg_sampling
            edge_label_index=(seed_edge_type, split_data[seed_edge_type].edge_label_index),
            edge_label=split_data[seed_edge_type].edge_label,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return mini_batch_loader

    def learn(self, loader, epochs):
        start = time.time()
        epochs_taken = 0
        loss_array = []
        val_auc_array = []
        val_loss_array = []
        earlystopping = EarlyStopping(patience=5, verbose=True, delta=0.001,
                                      path=f"{self.model_output}/state_dict_model.e{epochs}.pt", trace_func=print, save_model=False)
        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()  # ensuring clearing out the gradients before each validation loop

            total_loss = 0
            total_examples = 0
            torch.cuda.empty_cache()
            # train for loaders of all edge_types, e.g : train_loader['skill','to','team'], train_loader['member','to','team']
            for seed_edge_type in self.edge_types:
                print(f'epoch {epoch:03d} : batching for train_loader for seed_edge_type : {seed_edge_type}')
                for sampled_data in loader[seed_edge_type]:
                    self.optimizer.zero_grad()

                    sampled_data.to(self.device)
                    pred = self.model(sampled_data, seed_edge_type, self.is_directed)
                    # The ground_truth and the pred shapes should be 1-dimensional
                    # we squeeze them after generation
                    if (type(sampled_data) == HeteroData):
                        ground_truth = sampled_data[seed_edge_type].edge_label
                    else:
                        ground_truth = sampled_data.edge_label
                    loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += float(
                        loss) * pred.numel()  # each batch might not contain same number of predictions, so we normalize it using the individual number of preds each turn
                    total_examples += pred.numel()
            avg_loss = (total_loss / total_examples)
            if (epoch % 10 == 0):
                print(f"\n.............Epoch: {epoch:03d}, Loss: {avg_loss:.6f}.............\n")
            loss_array.append(avg_loss)

            self.optimizer.zero_grad()
            l1, l2 = self.eval(self.val_loader)
            val_loss_array.append(l1)
            val_auc_array.append(l2)

            epochs_taken += 1
            earlystopping(val_loss_array[-1], self.model)
            if earlystopping.early_stop:
                print(f"Early Stopping Triggered at epoch: {epoch}")
                break

        # plot the figure and save
        fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.png'
        self.plot_graph(torch.arange(1, epochs_taken + 1, 1), loss_array, val_loss_array, fig_output=fig_output)
        fig_output = f'{self.model_output}/{self.model_name}.{self.graph_type}.undir.{self.agg}.e{epochs}.ns{int(self.ns)}.b{self.b}.d{self.d}.val_auc_per_epoch.png'
        self.plot_graph(torch.arange(1, epochs_taken + 1, 1), val_auc_array, xlabel='Epochs', ylabel='Val AUC',
                   title=f'Validation AUC vs Epochs for Embedding Generation', fig_output=fig_output)
        print(f'\nit took {(time.time() - start) / 60} mins || {(time.time() - start) / 3600} hours to train the model\n')

    @torch.no_grad
    def eval(self, loader):
        preds = []
        ground_truths = []
        total_loss = 0
        total_examples = 0
        for seed_edge_type in self.edge_types:
            for sampled_data in loader[seed_edge_type]:
                sampled_data.to(self.device)
                tmp_pred = self.model(sampled_data, seed_edge_type, self.is_directed)
                if (type(sampled_data) == HeteroData):
                    # we have ground_truths per edge_label_index
                    tmp_ground_truth = sampled_data[seed_edge_type].edge_label
                else:
                    tmp_ground_truth = sampled_data.edge_label

                assert tmp_pred.shape == tmp_ground_truth.shape
                loss = F.binary_cross_entropy_with_logits(tmp_pred, tmp_ground_truth)
                total_loss += float(loss) * tmp_pred.numel()
                total_examples += tmp_pred.numel()

                preds.append(tmp_pred)
                ground_truths.append(tmp_ground_truth)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        loss = total_loss / total_examples
        auc = roc_auc_score(ground_truth, pred)
        print()
        print(f'Val loss : {loss:.6f}')
        print(f"Val AUC: {auc:.6f}\n")
        # print(f'................... ending eval...................\n')
        return loss, auc
