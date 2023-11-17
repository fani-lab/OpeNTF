import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import src.mdl.graph
from src.misc import data_handler

class GCN(src.mdl.graph.Graph):
    def __init__(self):
        super().__init__()
        self.init_child_variables()

    def init_model(self):
        # initialize the gcn_model object here
        self.gcn_model = GCN_Model(self.)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=True)  # Assuming training is always True
        x = self.conv2(x, edge_index)

        return x

    def init_child_variables(self):
        model_params = self.params['model'][self.model_name]['model_params']
        self.num_features = model_params['num_features']
        self.hidden_dim = model_params['hidden_dim']
        self.dropout = model_params['dropout']


    def run(self):

        # create custom graph data
        teams_graph = data_handler.create_custom_data(is_homogeneous = True)

        # Create an instance of the GCN model
        gcn_model = self.init_model(self.num_features, self.hidden_dim, self.embedding_dim)

        # Forward pass to get the graph embeddings
        graph_embeddings = gcn_model.forward(teams_graph)
        print("Graph Embeddings:\n", graph_embeddings)

class GCN_Model(torch.nn.Module):
    def __init__(self):
        super(GCN_Model, self).__init__()
        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.embedding_dim)

def main():
    gcn = GCN()
    gcn.run()

if __name__ == '__main__':
    main()