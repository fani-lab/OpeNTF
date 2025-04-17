from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
import torch.nn
from torch_geometric.data import HeteroData

class GS(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, source_node_emb, target_node_emb, edge_label_index) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_source = source_node_emb[edge_label_index[0]]
        edge_feat_target = target_node_emb[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge in the edge_label_index
        return (edge_feat_source * edge_feat_target).sum(dim=-1)


class Model(torch.nn.Module):
    # the data here is the global graph data that we loaded
    # need to verify whether in inductive training we pass the entire data info here
    def __init__(self, hidden_channels, data):
        super().__init__()
        if(type(data) == HeteroData):
            self.node_lin = []
            self.node_emb = []
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            # these two are generated such that both of them has the same shape and they can be added together
            for i, node_type in enumerate(node_types):
                self.node_lin.append(torch.nn.Linear(data[node_type].num_features, hidden_channels))
                self.node_emb.append(torch.nn.Embedding(data[node_type].num_nodes, hidden_channels))
        else:
            self.node_lin = torch.nn.Linear(data.num_features, hidden_channels)
            self.node_emb = torch.nn.Linear(data.num_nodes, hidden_channels)

        # Instantiate homogeneous gs:
        self.gs = GS(hidden_channels)

        if (type(data) == HeteroData):
            # Convert gs model into a heterogeneous variant:
            self.gs = to_hetero(self.gs, metadata=data.metadata())

        # instantiate the predictor class
        self.classifier = Classifier()

    def forward(self, data, is_directed) -> Tensor:

        if(type(data) == HeteroData):
            x_dict = {}
            self.x_dict = x_dict
            edge_types = data.edge_types if is_directed else data.edge_types[:(len(data.edge_types)) // 2]
            for i, node_type in enumerate(data.node_types):
                x_dict[node_type] = self.node_lin[i](data[node_type].x)
                x_dict = self.gs(x_dict, data.edge_index_dict)
                # this line is for batching mode
                # x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)
        else:
            x = self.node_lin(data.x)
            x = self.gs(x, data.edge_index)
            self.x = x
            # x_dict['node'] = self.node_lin(data.x) + self.node_emb(data.n_id)

        # `x_dict` holds embedding matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types


        # create an empty tensor and concatenate the preds afterwards
        preds = torch.empty(0)

        if (type(data) == HeteroData):
            # generate predictions per edge_label_index type
            # e.g: for edge_type 1, 2, 3
            # source_node_emb contains the embeddings of each node of the defined node_type
            for edge_type in edge_types:
                source_node_emb = x_dict[edge_type[0]]
                target_node_emb = x_dict[edge_type[2]]
                edge_label_index = data[edge_type].edge_label_index
                pred = self.classifier(source_node_emb, target_node_emb, edge_label_index)
                preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        else:
            pred = self.classifier(x, x, data.edge_label_index)
            # pred = self.classifier(x_dict['node'], x_dict['node'], data.edge_label_index)
            preds = torch.cat((preds, pred.unsqueeze(0)), dim = 1)
        return preds.squeeze(dim=0)