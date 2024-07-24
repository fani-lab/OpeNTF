'''

This implementation is inspired by an official implementation example from pyg
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_transductive.py

'''

import torch
import torch_geometric.data
from torch_geometric.nn import to_hetero
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch import nn as nn
from torch_geometric.nn import DeepGraphInfomax as DGI, GATConv
from decoder import Decoder



class LANT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads = 2)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0), device=x.device)], edge_index


class LANTModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dgi = DGI(
            hidden_channels=hidden_channels,
            encoder=to_hetero(LANT(hidden_channels, hidden_channels), metadata=data.metadata()),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )

        # self.dgi = to_hetero(self.dgi, metadata=data.metadata())

    def forward(self, data):
        pos_z, neg_z, summary = self.dgi(data.x_dict, data.edge_index_dict)
        return pos_z, neg_z, summary

if __name__ == '__main__':
    import pickle
    from torch_geometric.nn import to_hetero
    # Example usage:
    file = './../../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/sm.en.undir.add.data.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    model = LANTModel(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data)
        loss = model.dgi.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()