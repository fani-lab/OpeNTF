'''
This is the base implementation of heterogeneous dgi
with SAGEConv as the encoder. This implementation is for
a benchmark train run against actual LANT model with GATConv
as encoder
'''

import torch
import torch_geometric.data
from torch_geometric.nn import to_hetero
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch import nn as nn
from torch_geometric.nn import DeepGraphInfomax as DGI, SAGEConv, GATConv
from decoder import Decoder


# the LANT class with SAGEConv as encoder
class LANTbk(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)


    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


# the actual LANT class
class LANT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads = 2):
        super().__init__()
        self.conv = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads = heads)
        self.prelu = torch.nn.PReLU(hidden_channels * heads)


    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


# the encoder
class LANTModelBK(torch.nn.Module):
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
            summary=summary,
            corruption=corruption
        )

        # self.dgi = to_hetero(self.dgi, metadata=data.metadata())

    def forward(self, data):
        self.x_dict = {}
        if type(data) == HeteroData:
            for i, node_type in enumerate(data.node_types):
                self.x_dict[node_type] = self.node_lin[i](data[node_type].x)

        pos_z, neg_z, summary = self.dgi(self.x_dict, data.edge_index_dict)
        return pos_z, neg_z, summary


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

def create_toy_data():
    dummy_data = HeteroData()
    dummy_data['team'].x = torch.randn(10, 64)  # Example dummy feature matrix
    dummy_data['skill'].x = torch.randn(10, 64)
    dummy_data['member'].x = torch.randn(10, 64)
    dummy_data.edge_index_dict = {'team': torch.randint(0, 10, (2, 10)),
                                  'skill': torch.randint(0, 10, (2, 10)),
                                  'member': torch.randint(0, 10, (2, 10))}

    return data


'''
A dummy main class for model test
'''
if __name__ == '__main__':
    import pickle
    from torch_geometric.nn import to_hetero
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str)
    args = parser.parse_args()

    file = args.file

    with open(file, 'rb') as f:
        data = pickle.load(f)

    # data = create_toy_data()

    model = LANTModelBK(hidden_channels=64, data = data)
    # model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    losses = []
    n_epochs = 200
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data)
        # print(f"pos_z (team): {pos_z['team']}")
        # print(f"neg_z (team): {neg_z['team']}")

        loss = calculate_loss(model.dgi, pos_z, neg_z, summary)
        loss.backward()
        losses.append(loss.item())
        print(f'e : {epoch}, l : {loss}')
        optimizer.step()

    import matplotlib.pyplot as plt
    # Plotting the loss
    plt.plot(range(n_epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.show()