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
from torch_geometric.nn import DeepGraphInfomax as DGI, GATConv, GCNConv, SAGEConv
from decoder import Decoder



class LANT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads = 2):
        super().__init__()
        self.conv = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads = heads)
        self.prelu = torch.nn.PReLU(hidden_channels * heads)


    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x

# utils for LANT
def corruption(x_dict, edge_index_dict):
    corrupted_x_dict = {key: x[torch.randperm(x.size(0), device=x.device)] for key, x in x_dict.items()}
    return corrupted_x_dict, edge_index_dict

def summary(z_dict, *args, **kwargs):
    # Compute the mean for each node type
    summary_dict = {key: z.mean(dim=0, keepdim=True) for key, z in z_dict.items()}
    # Aggregate the means
    return torch.sigmoid(torch.cat(list(summary_dict.values()), dim=0).mean(dim=0))


# to calculate the losses based on the fact of pos_z, neg_z and summary being dicts
def calculate_loss(dgi, pos_z_dict, neg_z_dict, summary):
    total_loss = 0.0

    # Iterate through all node types
    for node_type in pos_z_dict.keys():
        pos_z = pos_z_dict[node_type]
        neg_z = neg_z_dict[node_type]

        # Calculate the loss for this node type
        loss = dgi.loss(pos_z, neg_z, summary)
        # print(f'----Loss for node_type {node_type}: {loss.item()}')

        # Aggregate the losses, here we are summing them
        total_loss += loss

    # You might want to average the loss if you are using mini-batches
    average_loss = total_loss / len(pos_z_dict)

    return average_loss
