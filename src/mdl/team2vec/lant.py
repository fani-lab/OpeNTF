'''

This implementation is inspired by an official implementation example from pyg
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/infomax_transductive.py

'''

import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import to_hetero
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch import nn as nn
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.metrics import roc_auc_score


class LANT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads = 4):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads = heads)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads = 1)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        return x

# utils for LANT
def corruption(x_dict, edge_index_dict):
    corrupted_x_dict = {key: x[torch.randperm(x.size(0), device=x.device)] for key, x in x_dict.items()}
    return corrupted_x_dict, edge_index_dict

def summary_function(z_dict, *args, **kwargs):
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


# calculate loss and auc for data of only one edge_type
def calculate_link_prediction_loss_auc(pos_z_dict, edge_type, data):

    all_edge_preds = []
    all_edge_labels = []

    bce_loss = nn.BCEWithLogitsLoss()
    pos_z_src = pos_z_dict[edge_type[0]]
    pos_z_dst = pos_z_dict[edge_type[-1]]

    edge_label_index = data[edge_type].edge_label_index
    edge_labels = data[edge_type].edge_label.float()  # Assuming edge_label is a tensor

    edge_preds = []
    for i in range(edge_label_index.shape[1]):
        src, dst = edge_label_index[:, i]
        edge_pred = (pos_z_src[src] * pos_z_dst[dst]).sum().unsqueeze(0)
        edge_preds.append(edge_pred)

    edge_preds = torch.cat(edge_preds)

    # Calculate BCE loss for link prediction
    lp_loss = bce_loss(edge_preds, edge_labels)

    # Collect all predictions and labels for AUC calculation
    all_edge_preds.append(edge_preds.detach().cpu().numpy())
    all_edge_labels.append(edge_labels.detach().cpu().numpy())

    # Concatenate all predictions and labels
    all_edge_preds = np.concatenate(all_edge_preds)
    all_edge_labels = np.concatenate(all_edge_labels)

    # Calculate AUC score
    auc = roc_auc_score(all_edge_labels, all_edge_preds)

    return lp_loss, auc
