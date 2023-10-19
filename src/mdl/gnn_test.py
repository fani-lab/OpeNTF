import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

def load_dataset():
    # Load a graph dataset
    # dataset = KarateClub()
    dataset = Planetoid(root = '~/tmp/Cora', name = 'Cora')


    return dataset

def explore_dataset(dataset):
    data = dataset[0]
    print(f'Dataset: {dataset}:')
    print('======================')

    print(f'dimension of the data : {len(data)}')

    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    # print(f'Range of training nodes : {data.train_mask}')
    print(f'Type of data : {type(data)}')
    print(f'')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

# visualize the data
def visualize_dataset(data):
    # G = to_networkx(data, to_undirected=True)
    # nx.draw(G)
    pass

def create_model(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e4)
    return model, data, optimizer
    pass

def train_model(model, data, optimizer):
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model


def evaluate_model(model, data):
    model.eval()
    pred = model(data).argmax(dim = 1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')



class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim = 1)


def main():
    dataset = load_dataset()
    data = dataset[0]
    explore_dataset(dataset)
    # visualize_dataset(data)
    model, data, optimizer = create_model(dataset)
    train_model(model, data, optimizer)
    evaluate_model(model, data)

if __name__ == "__main__":
    main()