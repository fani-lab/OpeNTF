'''
We create a separate encoder class for lant due to the difference
in training and loss calculation mechanism in lants
'''

import torch
import torch_geometric.data
from torch_geometric.nn import to_hetero
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch import nn as nn
from torch_geometric.nn import DeepGraphInfomax as DGI, GATConv
from lant import LANT
from decoder import Decoder

'''
This is functionally a bit different from the Encoder class of encoder.py
'''
class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.heads = 2

        '''
        this section of linear transformation is needed to add random feature matrix to the initial nodes with 1 dimensional features
        (as our data doesnt have its own node features. But the intricate part here is that we at first linear transform the node features
        into our desired dimensions = hidden_channels. But then before feeding to the dgi encoder, we have to divide it by the number of 
        attention heads of the GATConv layer. Because after feeding into the GAT encoder of the dgi, the output channels will multiply by 
        the number of heads and so the initial dimension would be increased. To counter that, we do the early division by heads before 
        feeding to the GAT encoder)
        '''
        if (type(data) == HeteroData):
            self.node_lin = []
            # self.node_emb = []
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            # these two are generated such that both of them has the same shape and they can be added together
            for i, node_type in enumerate(node_types):
                if (data.is_cuda):
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels).cuda())
                    # self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels).cuda())
                else:
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels))
                    # self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels))

        self.hidden_channels = hidden_channels // self.heads # the attention heads cause the dimension to double inside the model

        self.encoder = to_hetero(LANT(self.hidden_channels, self.hidden_channels, heads = self.heads), metadata=data.metadata())
        # self.encoder = to_hetero(LANTbk(hidden_channels, hidden_channels), metadata=data.metadata())

        self.dgi = DGI(
            hidden_channels=self.hidden_channels * self.heads,
            encoder=self.encoder,
            summary=self.encoder.summary,
            corruption=self.encoder.corruption
        )

        # self.dgi = to_hetero(self.dgi, metadata=data.metadata())

    def forward(self, data, seed_edge_type, is_directed, emb=False) -> Tensor:
        if (type(data) == HeteroData):
            self.x_dict = {}
            for i, node_type in enumerate(data.node_types):
                self.x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)

            # `x_dict` holds embedding matrices of all node types
            # `edge_index_dict` holds all edge indices of all edge types
            self.x_dict = self.model(self.x_dict, data.edge_index_dict)
            if emb: return self.x_dict
        else:
            # for batching mode and homogeneous graphs, this line should be tested by appending node_emb part
            # e.g : if self.b : x_dict['node'] = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.model(x, data.edge_index)
            self.x = x
            if emb: return self.x

        pos_z, neg_z, summary = self.dgi(self.x_dict, data.edge_index_dict)
        return pos_z, neg_z, summary

    def learn(self):
        pass

    def eval(self):
        pass