import numpy as np, math, os, itertools, pickle
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, HeteroData

from team2vec import Team2Vec

class Gnn(Team2Vec):

    def __init__(self, teamsvecs, indexes, settings, output):
        super().__init__(teamsvecs, indexes, settings, output)
    def create(self, file):
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.remove_self_loops

        if not isinstance(self.settings['edge_types'][0], list):#homo
            teams = self.teamsvecs[self.settings['edge_types'][0]] #TODO: if node_type == 'team'
            edges = []
            for i, row in enumerate(tqdm(teams)):
                for t in itertools.combinations(row.nonzero()[1], 2): edges += [t]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() #[1,2][1,2] >> [1,1][2,2]
            nodes = torch.tensor([[0]] * teams.shape[1], dtype=torch.float)
            self.data = Data(x=nodes, edge_index=edge_index, edge_attr=torch.tensor([1] * len(edges), dtype=torch.long))
        else:
            self.data = HeteroData()
            node_types = set()
            #edges
            for edge_type in self.settings['edge_types'][0]:
                teams = self.teamsvecs[edge_type[0]]
                edges = []
                for i, row1 in enumerate(tqdm(teams)):
                    row2 = self.teamsvecs[edge_type[2]][i] if edge_type[2] != 'team' else [i]
                    for t in itertools.product(row1.nonzero()[1], row2.nonzero()[1] if edge_type[2] != 'team' else row2): edges += [t]
                self.data[edge_type].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                self.data[edge_type].edge_attr = torch.tensor([1] * len(edges), dtype=torch.long)
                node_types = node_types.union({edge_type[0], edge_type[2]})
            #nodes
            for node_type in node_types: self.data[node_type].x = torch.tensor([[0]] * (self.teamsvecs[node_type].shape[1] if node_type != 'team' else self.teamsvecs['id'].shape[0]), dtype=torch.float)

        if not self.settings['dir']:
            import torch_geometric.transforms as T
            transform = T.ToUndirected()
            self.data = transform(self.data)

        if self.settings['dup_edge']:
            import torch_geometric.transforms as T
            transform = T.RemoveDuplicatedEdges(key=["edge_attr"], reduce=self.settings['dup_edge'])
            self.data = transform(self.data)

        self.data.validate(raise_on_error=True)
        with open(file, 'wb') as f: pickle.dump(self.data, f)
        return self.data

    def plot(self, x, y, output_filepath = None):
        mx = max(y)
        mn = min(y)
        threshold = (mx - mn) // 10

        plt.figure()
        plt.ylabel('Loss')
        plt.ylim(mn - threshold, mx + threshold)
        plt.xlabel('Epochs')
        plt.xlim(0, len(x))
        plt.plot(x, y)
        plt.legend('This is a legend')
        # save the figure before showing the plot
        # because after plt.show() another blank figure is created
        if(output_filepath):
            print(f'Saving plot in {output_filepath}')
            plt.savefig(output_filepath)

        plt.show()

    def plot_homogeneous_graph(self, teams_graph):
        G = torch_geometric.utils.to_networkx(teams_graph, to_undirected=True)
        nx.draw(G, with_labels=True)
