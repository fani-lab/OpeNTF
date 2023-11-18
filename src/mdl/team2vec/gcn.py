import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import src.mdl.gnn.graph

class GCN(src.mdl.gnn.graph.Graph):
    def __init__(self):
        super().__init__()

    def init_model(self):
        self.gcn_model = GCN_Model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()


    def run(self):
        teams_graph = data_handler.create_custom_data(is_homogeneous = True)
        gcn_model = GCN_Model(self)
        self.init_model(self.num_features, self.hidden_dim, self.embedding_dim)
        graph_embeddings = gcn_model.forward(teams_graph)
        print("Graph Embeddings:\n", graph_embeddings)

class GCN_Model(torch.nn.Module):
    def __init__(self, gcn):
        super(GCN_Model, self).__init__()
        # to access the fetaures of graph > gcn heirarchy
        self.gcn = gcn
        self.conv1 = GCNConv(self.gcn.num_features, self.gcn.hidden_dim)
        self.conv2 = GCNConv(self.gcn.hidden_dim, self.gcn.embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.gcn.dropout, training = self.gcn.training)  # Assuming training is always True
        x = self.conv2(x, edge_index)
        return x
