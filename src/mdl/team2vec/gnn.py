import numpy as np, networkx as nx, math, os, itertools
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch_geometric.data import Data, HeteroData

from team2vec import Team2Vec

class Gnn(Team2Vec):

    def __init__(self, teamsvecs, indexes, model, settings, output):
        super(Team2Vec, self).__init__(teamsvecs, indexes, output, settings=settings)
        self.model = model
    def create(self, file):
        if not isinstance(self.settings['edge_types'], list):#homo
            teams = teamsvecs[ntypes[0]]
            for i, row in enumerate(tqdm(teamsvecs[ntypes[0]])):
                # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RemoveDuplicatedEdges.html#torch_geometric.transforms.RemoveDuplicatedEdges
                edges = []
                for t in itertools.combinations(row.nonzero(), 2): edges += [list(t), list(reversed(t))]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() #[1,2][1,2] >> [1,1][2,2]
            nodes = torch.tensor([[0]] * teams.shape[1], dtype=torch.float)
            data = Data(x=nodes, edge_index=edge_index)
        else:
            data = HeteroData()
            edges = []
            for ntype in ntypes: data[ntype].x = torch.tensor([[0]] * teams.shape[1 if ntype != 'team' else 0], dtype=torch.float)
            # data['x'].x = torch.tensor([[0]] * 3, dtype=torch.float)
            # data['y'].x = torch.tensor([[0]] * 4, dtype=torch.float)

            for edge_type in self.settings['edge_types']:
                for i, row1 in enumerate(tqdm(teamsvecs[edge_type[0]])):
                    edges = []; rev_edges = []
                    row2 = teamsvecs[edge_type[2]] if edge_type[2] != 'team' else [1] * teamsvecs[edge_type[0]].shape[0]
                    for t in itertools.product(row1.nonzero(), row2.nonzero()):
                        edges += list(t)
                        rev_edges += list(reversed(t))
                    data[edge_type].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    data[reversed(edge_type)].edge_index = torch.tensor(rev_edges, dtype=torch.long).t().contiguous()
            # data[('x', '-', 'y')].edge_index = torch.tensor([[1, 1], [1, 2], [2, 3]], dtype=torch.long).t().contiguous()
            # data[('y', '-', 'x')].edge_index = torch.tensor([[1, 1], [1, 2], [3, 2]], dtype=torch.long).t().contiguous()
            # data[('x', '-', 'x')].edge_index = torch.tensor([[1, 1], [1, 2], [2, 2]], dtype=torch.long).t().contiguous()
            # data[('y', '-', 'y')].edge_index = torch.tensor([[1, 1], [1, 2], [3, 3]], dtype=torch.long).t().contiguous()

        data.validate(raise_on_error=True)
        with open(file, 'wb') as f: pickle.dump(data, f)
        return data

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
