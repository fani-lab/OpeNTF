from torch_geometric.nn import to_hetero
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch_geometric.data import Data, HeteroData
import torch

from gs import GS
from gat import GAT
from gatv2 import GATv2
from han import HAN
from gin import GIN
from gine import GINE
from decoder import Decoder


class Encoder(nn.Module):
    # the data here is the global graph data that we loaded
    # need to verify whether in inductive training we pass the entire data info here
    def __init__(self, hidden_channels, data, model_name):
        super().__init__()

        self.model_name = model_name
        # Define a dictionary that maps model names to class constructors
        model_classes = {
            'gs': GS,
            'gat': GAT,
            'gatv2': GATv2,
            'gin': GIN,
            'gine': GINE
            # HAN is excluded because it has a separate case
        }

        if(type(data) == HeteroData):
            self.node_lin = []
            self.node_emb = []
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            # these two are generated such that both of them has the same shape and they can be added together
            for i, node_type in enumerate(node_types):
                if (data.is_cuda):
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels).cuda())
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels).cuda())
                else:
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels))
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels))
        else:
            if (data.is_cuda):
                self.node_lin = nn.Linear(data.num_features, hidden_channels).cuda()
                self.node_emb = nn.Linear(data.num_nodes, hidden_channels).cuda()
            else:
                self.node_lin = nn.Linear(data.num_features, hidden_channels)
                self.node_emb = nn.Linear(data.num_nodes, hidden_channels)

        # Instantiate homogeneous model:
        if self.model_name == 'han':
            self.model = HAN(hidden_channels, data.metadata())
        else:
            self.model = model_classes[model_name](hidden_channels)

        if (type(data) == HeteroData and model_name != 'han'):
            # Convert the homogeneous (if applicable) model into a heterogeneous variant:
            self.model = to_hetero(self.model, metadata=data.metadata())

        # instantiate the predictor class
        self.decoder = Decoder()

    # emb = True enables to just output the embedding generated by the model after
    # the entire training phase has been done
    def forward(self, data, seed_edge_type, is_directed, emb = False) -> Tensor:
        if(type(data) == HeteroData):
            self.x_dict = {}
            for i, node_type in enumerate(data.node_types):
                self.x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)
            if self.model_name == 'gine':
                self.edge_attr_dict = {}
                for edge_type in data.edge_types:
                    self.edge_attr_dict[edge_type] = data[edge_type].edge_attr.view(-1, 1).float()

                self.x_dict = self.model(self.x_dict, data.edge_index_dict, self.edge_attr_dict)
            else:
                # `x_dict` holds embedding matrices of all node types
                # `edge_index_dict` holds all edge indices of all edge types
                self.x_dict = self.model(self.x_dict, data.edge_index_dict)
            if emb : return self.x_dict
        else:
            # for batching mode and homogeneous graphs, this line should be tested by appending node_emb part
            # e.g : if self.b : x_dict['node'] = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.node_lin(data.x) + self.node_emb(data.n_id)
            x = self.model(x, data.edge_index)
            self.x = x
            if emb : return self.x

        # create an empty tensor and concatenate the preds afterwards
        preds = torch.empty(0).to(device = 'cuda' if data.is_cuda else 'cpu')

        if (type(data) == HeteroData):
            # generate predictions only on the edge_label_index of seed_edge_type
            # e.g : for seed_edge_type ['skill','to','team'], we choose to only predict from data['skill','to','team'].edge_label_index
            # source_node_emb contains the embeddings of each node of the defined node_type
            source_node_emb = self.x_dict[seed_edge_type[0]]
            target_node_emb = self.x_dict[seed_edge_type[2]]
            pred = self.decoder(source_node_emb, target_node_emb, data[seed_edge_type].edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        else:
            pred = self.decoder(x, x, data.edge_label_index)
            # pred = self.classifier(x_dict['node'], x_dict['node'], data.edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        return preds.squeeze(dim=0)