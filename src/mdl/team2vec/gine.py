import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINEConv


class GINE(torch.nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()), edge_dim = 1, train_eps = True)

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)

        return x
