# import classes from opentf
import param
import sys,os, json
from json import JSONEncoder
from cmn.team import Team
from cmn.author import Author
from cmn.publication import Publication

from torch_geometric.nn import Node2Vec, MetaPath2Vec
from torch_geometric.data import Data
import torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid

def create_data(x, edge_index, y = None):
    # x = torch.tensor([-1, 0, 1])
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    # y = torch.tensor([2, 4, 1])

    # the data is only for generating the embedding without any target labels
    if not y :
        data = Data(x = x, edge_index = edge_index)
    else:
        data = Data(x = x, edge_index = edge_index, y = y)
    return data

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

## main ##
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # create a sample data to test
    # this data will have 3 nodes 0, 1 and 2 with 0-1, 1-2 but no 0-2 edge
    # the similarity should be between 0-1, 1-2 but 0-2 should be different from one another
    # data = create_data(torch.tensor([[0], [1], [2]], dtype=torch.float), torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)).to(device)

    data = Planetoid('./data/Planetoid', name='Cora')[0]

    model = Node2Vec(
        data.edge_index,
        embedding_dim = 128,
        walks_per_node = 10,
        walk_length = 20,
        context_size = 10,
        p = 1.0,
        q = 1.0,
        num_negative_samples = 1
    ).to(device)

    print(type(model))
    print(model)
    print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    try :
        pos_rw, neg_rw = next(iter(loader))
    except EOFError:
        print('EOFError')

    for epoch in range(1, 100):
        loss = train()
        print(f'epoch = {epoch : 02d}, loss = {loss : .4f}')




