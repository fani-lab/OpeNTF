import numpy as np, math, os, itertools, pickle, time, json
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, HeteroData

from team2vec import Team2Vec

class Gnn(Team2Vec):

    def __init__(self, teamsvecs, indexes, settings, output):
        super().__init__(teamsvecs, indexes, settings, output)

        self.loader = None
        self.optimizer = None
        self.device = 'cuda'
        if not os.path.isdir(self.output): os.makedirs(self.output)
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

    def train(self, epochs, save_per_epoch=False):
        self.model.to(self.device)
        model_output = f'{self.output}/{self.model_name}'
        if not os.path.isdir(model_output): os.makedirs(model_output)
        train_loss_values = []
        valid_loss_values = []
        t_start_time = time.time()
        for epoch in range(epochs):
            e_start_time = time.time()
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))

                #TODO: we can inject opentf here and add the loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss = total_loss / len(self.loader)
            train_loss_values.append(loss)

            # TODO: directly launch a pretrained model
            acc = self.valid()
            valid_loss_values.append(acc)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Time: {time.time() - e_start_time: 0.4f}')
            if save_per_epoch: torch.save(self.model.state_dict(), f'{model_output}/gnn_model.e{epoch}.pt', pickle_protocol=4)
        torch.save(self.model.state_dict(), f'{model_output}/gnn_model.pt', pickle_protocol=4)
        #to load later by: self.model.load_state_dict(torch.load(f'{self.output}/gnn_model.pt'))

        print(f'It took {time.time() - t_start_time} to train the model.')
        with open(f'{model_output}/train_valid_loss.json', 'w') as outfile: json.dump({'train': train_loss_values, 'valid': valid_loss_values}, outfile)
        plt.figure()
        plt.plot(train_loss_values, label='Training Loss')
        plt.plot(valid_loss_values, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title(f'Training and Validation Losses per Epoch')
        plt.savefig(f'{model_output}/train_valid_loss.png', dpi=100, bbox_inches='tight')
        plt.show()

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



