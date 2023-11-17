import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN:
    def __init__(self, num_features, hidden_dim, embedding_dim=3, dropout=0.5):
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=True)  # Assuming training is always True
        x = self.conv2(x, edge_index)

        return x

def run():
    # Create a simple homogeneous graph with 4 nodes and no features
    x = torch.tensor([[0], [1], [2], [3]], dtype = torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2],
                               [1, 0, 3, 2, 2, 0]], dtype=torch.long)
    graph_data = Data(x = x, edge_index=edge_index)

    # Specify the number of input features, hidden dimensions, and embedding dimension
    num_features = 1  # Since there are no features, set it to 1
    hidden_dim = 16
    embedding_dim = 3

    # Create an instance of the GCN model
    gcn_model = GCN(num_features, hidden_dim, embedding_dim)

    # Forward pass to get the graph embeddings
    graph_embeddings = gcn_model.forward(graph_data)

    print("Graph Embeddings:\n", graph_embeddings)

if __name__ == '__main__':
    run()