# import classes from opentf
import torch_geometric.utils

import param
import sys,os, json
from json import JSONEncoder
from cmn.team import Team
from cmn.author import Author
from cmn.publication import Publication

from torch_geometric.nn import Node2Vec, MetaPath2Vec
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
# import torch.utils.data.dataloader


### Hossein Fani ###
# def create_data(teams, nodetype=['member', 'skill']):
#     for i in nodetype:
#         teams[i]# the rows of type i
#
#
#     members, skills, teams = teams[nodetype[i]], teams['skill'], teams['id']
#     if nodetype == 'expert':
#         # you need to create a graph of _, graphsize = members.shape
#         for row in members:
#             row.data ==> column idx of experts who are connected to each other in this row (team)
#             create the edges between the idxes
#
#     # x = torch.tensor([-1, 0, 1])
#     # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
#     # y = torch.tensor([2, 4, 1])
#
#     # the data is only for generating the embedding without any target labels
#     if not y :
#         data = Data(x = x, edge_index = edge_index)
#     else:
#         data = Data(x = x, edge_index = edge_index, y = y)
#     return data

def create_data_naive(x, edge_index, y = None):
    # x = torch.tensor([-1, 0, 1])
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    # y = torch.tensor([2, 4, 1])

    # the data is only for generating the embedding without any target labels
    if not y :
        data = Data(x = x, edge_index = edge_index)
    else:
        data = Data(x = x, edge_index = edge_index, y = y)
    return data

# initialize the model for training
def init():
    model = Node2Vec(
        data.edge_index,
        embedding_dim = 3,
        walks_per_node = 10,
        walk_length = 4,
        context_size = 2,
        p = 1.0,
        q = 1.0,
        num_negative_samples = 1
    ).to(device)

    print(type(model))
    print(model)
    print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    # the "TypeError: cannot pickle 'PyCapsule' object" error resolved after removing num_workers from the
    # set of arguments in the following model.loader() call
    num_workers = 4 if sys.platform == 'linux' else 0
    loader = model.loader(batch_size=50, shuffle=True, num_workers = num_workers)
    try :
        # on Cora dataset having x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708],
        # the next(iter(loader)) produces a tuple of two tensors pos_rw and neg_rw
        # each has a dimension of (14080, 10)
        pos_rw, neg_rw = next(iter(loader))
    except EOFError:
        print('EOFError')
    return model, loader, optimizer

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def plot_graph(team_graph):
    # make sure that the data given is a Data type object
    assert type(team_graph) == torch_geometric.data.Data

    G = torch_geometric.utils.to_networkx(team_graph, to_undirected = True)
    nx.draw(G, with_labels = True)

def plot(x, y):
    mx = max(y)
    mn = min(y)
    threshold = (mx - mn) // 10

    plt.figure()
    plt.ylabel('Loss')
    plt.ylim(mn - threshold, mx + threshold)
    plt.xlabel('Epochs')
    plt.xlim(0, len(x))
    plt.plot(x, y)
    plt.legend()
    plt.show()

# normalizr an np_array into a range of 0-1
def normalize(np_array):
    mx = np.max(np_array)
    mn = np.min(np_array)

    print(f'\nNormalize()\n')
    print(f'Before :')
    print(f'{np_array}')

    np_array = (np_array - mn) / (mx - mn)

    print(f'After :')
    print(f'{np_array}')
    print('')

    return np_array

## main ##
if __name__ == "__main__":

    # declare necessary variables
    # the respective heirarchy names needed for the output
    model_name = 'node2vec'
    parent_name = 'gnn_emb'
    dataset_name = 'custom'
    # the output_path for the testing logs
    output_path = f'../../output/{dataset_name}/{parent_name}/{model_name}/'
    os.makedirs(output_path, exist_ok = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print()
    print(f'The current device is {device}')
    print()

    # create a sample data to test
    # this data will have 3 nodes 0, 1 and 2 with 0-1, 1-2 but no 0-2 edge
    # the similarity should be between 0-1, 1-2 but 0-2 should be different from one another
    data = create_data_naive(torch.tensor([[0], [1], [2]], dtype=torch.float), torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)).to(device)
    # data = Planetoid('./data/Planetoid', name='Cora')[0]

    # plot the graph data
    plot_graph(data)

    ### Hossein Fani
    # teams = {}
    # teams['skill'] = None
    # teams['location'] = None
    # teams['teams'] = None
    # teams['members'] = scipy.sparse.lil([[1,1,0],[0, 1, 1]])
    ###

    # we test for some set of epochs and generate output files based on them
    num_epochs = [100, 200, 500, 1000, 1500, 2000]
    for max_epochs in num_epochs:
        # initialize the model everytime
        model, loader, optimizer = init()

        # we keep track of losses in a set of epochs
        losses = []

        # the data file that needs to log our performances
        output_filename = 'performance_log_'
        output_filename += f'{max_epochs}_epochs.txt'
        output_filename_with_path = output_path + output_filename
        with open(output_filename_with_path, 'w') as outfile:
            line = f'Graph : \n\n' \
                   f'x = {data.x}\n' \
                   f'edge_index = {data.edge_index}\n\n' \
                   f'---------------------------------\n' \
                   f'---------------------------------\n' \
                   f'\nNumber of Epochs : {max_epochs}\n' \
                   f'---------------------------------\n'
            for epoch in range(1, max_epochs + 1):
                loss = train()
                losses.append(loss)
                if(epoch % 10 == 0):
                    print(f'Epoch = {epoch : 02d}, loss = {loss : .4f}')

                    # the model() gives all the weights and biases of the model currently
                    # the detach() enables this result to require no gradient
                    # and then we convert the tensor to numpy array
                    weights = model().detach().numpy()
                    weights = normalize(weights)
                    weights = np.around(weights, 2)

                    print('\nembedding : \n')
                    print(weights)

                    # lines to write to file
                    line += f'Epoch : {epoch}\n'
                    line += f'--------------------------\n'
                    line += f'Node ----- Embedding -----\n\n'
                    for i, weights_per_node in enumerate(weights):
                        print(weights_per_node)
                        line += f'{i:2} : {weights_per_node}\n'
                    line += f'--------------------------\n\n'
            # write to file
            outfile.write(line)

        # plotting losses vs epochs
        plot(range(1, len(losses) + 1), losses)
