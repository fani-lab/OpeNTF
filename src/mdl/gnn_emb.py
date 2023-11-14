# import classes from opentf
import torch_geometric.utils

import param
import sys,os, json
from json import JSONEncoder
from cmn.team import Team
from cmn.author import Author
from cmn.publication import Publication
from misc import data_handler

from torch_geometric.nn import Node2Vec, MetaPath2Vec
from torch_geometric.data import Data, HeteroData
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

# create any custom data here and delete the example when not needed
def create_custom_data():
    teamsvecs = data_handler.read_data(preprocessed_datapath)
    # here 'id' refers to the sparse_matrix of team
    teams_graph = create_hetero_data(teamsvecs, ['id', 'skill', 'member'], \
                                     [['skill', 'member'], ['member', 'skill'],['id', 'member'], ['member', 'id']], \
                                     preprocessed_datapath)

# create heterogeneous graph from
# teamsvecs = the information about the teams from which we have to infer the edges
# node_types = the types of nodes provided
# edge_types = the types of edges provided
def create_hetero_data(teamsvecs, node_types, edge_types, output_filepath):
    print('--------------------------------')
    print('create_hetero_data()')
    print('--------------------------------')
    print()

    teams_graph = HeteroData()

    print(f'Node types are : {node_types}')
    print(f'Edge types are : {edge_types}')

    # add the node types and create edge_indices for each edge_types
    # each node_types should have an x
    set_x = {}
    set_edge_index = {}
    for node_type in node_types:
        # the node numbers are indexed from 0 - the number of columns in each type
        # but for the id (teams) type, it is from 0 - the number of rows in this type
        start = 0
        end = teamsvecs['id'].shape[0] if node_type == 'id' else teamsvecs[node_type].shape[1]
        set_x[node_type] = np.arange(start, end)
    for edge_type in edge_types:
        key1, key2 = edge_type[0], edge_type[1]
        set_edge_index[key1, key2] = [[],[]]

    visited_index = {}
    # now for each type of edge_indices, populate the edge_index
    for edge_type in edge_types:
        # e.g : edge_types = [['skill', 'member'], ['id','member']]
        # these keys are for convenience in usage afterwards
        key1, key2 = edge_type[0], edge_type[1]
        reverse_edge_type = [key2, key1]
        # this dict_key adds one more key for the edge_type in the visited_index dict
        dict_key = str(key1 + key2)
        reverse_dict_key = str(key2 + key1)
        print(f'edge_type = {edge_type}, reverse_edge_type = {reverse_edge_type}')

        # let node_type N1 = 'skill', node_type N2 = 'member'
        # traverse N1 nodes row by row (each row represents the nodes in a single team)
        # for each node in N1, search for the nodes in N2 in the same row
        # if not visited[node_in_N1, node_in_N2], then visit them and create an edge_pair
        # repeat for every single row that contains the nonzero elements

        # lil_matrix for two node_type : n1, n2
        n1 = teamsvecs[key1]
        n2 = teamsvecs[key2]

        if(key1 != 'id' and key2 != 'id'):
            # for the node type 'id', we need to handle the edge_index population differently

            for row_id in range(n1.shape[0]):
                # the col_ids of the node_type1
                cols1 = n1.getrow(row_id).nonzero()[1]
                for col1 in cols1:
                    # for each col_id in n1, now we have to create pairs with the same row elements in n2
                    cols2 = n2.getrow(row_id).nonzero()[1]
                    for col2 in cols2:
                        if((not visited_index.get((col1, col2, dict_key), None)) and (not visited_index.get((col2, col1, reverse_dict_key), None))):
                            # mark the pair visited
                            visited_index[col1, col2, dict_key] = 1
                            visited_index[col2, col1, reverse_dict_key] = 1

                            # create 2 sets of edges between col1 and col2
                            # from col1 to col2, it should be edge_type 1,
                            # from col2 to col1, it should be reverse_edge_type
                            print(f'edge_index appended with edge pairs between n1 node {col1} and n2 node {col2}')
                            set_edge_index[key1, key2][0].append(col1)
                            set_edge_index[key1, key2][1].append(col2)
                            set_edge_index[key2, key1][0].append(col2)
                            set_edge_index[key2, key1][1].append(col1)
                            print(f'updated edge_index = {set_edge_index[key1, key2]}')
                            print(f'updated reverse_edge_index = {set_edge_index[key2, key1]}')
        elif(key2 == 'id'):
            # we traverse only one set of node pairs where
            # the second one is 'id'
            # to cover the edge type for both team-to-X and X-to-team
            for row_id in range(n1.shape[0]):
                # the col_ids of the node_type1
                cols1 = n1.getrow(row_id).nonzero()[1]
                # for each col_id in n1, now we have to create pairs with just the
                # row number which is the 'team' of that node_type1
                for col1 in cols1:
                    if((not visited_index.get((col1, row_id, dict_key), None)) and (not visited_index.get((row_id, col1, reverse_dict_key), None))):
                        # mark the pair visited
                        visited_index[col1, row_id, dict_key] = 1
                        visited_index[row_id, col1, reverse_dict_key] = 1

                        set_edge_index[key1, key2][0].append(col1)
                        set_edge_index[key1, key2][1].append(row_id)
                        set_edge_index[key2, key1][0].append(col2)
                        set_edge_index[key2, key1][1].append(row_id)
                        print(f'updated edge_index = {set_edge_index[key1, key2]}')
                        print(f'updated reverse_edge_index = {set_edge_index[key2, key1]}')

    # convert the collected data into torch for HeteroData
    for node_type in node_types:
        teams_graph[node_type].x = torch.tensor(set_x[node_type], dtype = torch.float64)
    for edge_type in edge_types:
        key1, key2 = edge_type[0], edge_type[1]
        teams_graph[key1, key2].edge_index = torch.tensor(np.array(set_edge_index[key1, key2]), dtype = torch.long)

    print()
    print('----------------------------------------------------')
    print(f'teams_graph = {teams_graph}')
    print()

    # can this return type be generalized for both homo and hetero graphs?
    return teams_graph

# the method to add edges of a particular edge_type to the edge_index
def add_edges(is_hetero):
    # mark the pair visited
    visited_index[col1, col2, dict_key] = 1
    visited_index[col2, col1, reverse_dict_key] = 1

    # create 2 sets of edges between col1 and col2
    # from col1 to col2, it should be edge_type 1,
    # from col2 to col1, it should be reverse_edge_type
    print(f'edge_index appended with edge pairs between n1 node {col1} and n2 node {col2}')
    set_edge_index[key1, key2][0].append(col1)
    set_edge_index[key1, key2][1].append(col2)
    set_edge_index[key2, key1][0].append(col2)
    set_edge_index[key2, key1][1].append(col1)
    print(f'updated edge_index = {set_edge_index[key1, key2]}')
    print(f'updated reverse_edge_index = {set_edge_index[key2, key1]}')

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

def plot_graph(teams_graph):
    # make sure that the data given is a Data type object
    assert type(teams_graph) == torch_geometric.data.Data

    G = torch_geometric.utils.to_networkx(teams_graph, to_undirected = True)
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
    dataset_version = 'toy.dblp.v12.json'
    filename = 'teamsvecs.pkl'
    graph_filename = 'teams_graph.pkl'
    # domains = param.settings['data']['domain']
    domain = 'dblp'

    # the path to teams_graph.pkl file
    teams_graph_input_filepath = f'../../data/graph/{domain}/{dataset_version}/{model_name}/{graph_filename}'
    teams_graph_output_filepath = f'../../data/graph/{domain}/{dataset_version}/{model_name}/{graph_filename}'
    preprocessed_datapath = f'../../data/preprocessed/{domain}/{dataset_version}/{filename}'
    # the output_path for the testing preprocess steps logs
    output_path = f'../../output/{dataset_name}/{parent_name}/{model_name}/'
    os.makedirs(output_path, exist_ok = True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print()
    print(f'The current device is {device}')
    print()

    ### Custom Section ###
    # create a sample data to test, change the name to data once testing is done
    # node_types, edge_types
    create_custom_data()

    # this data will have 3 nodes 0, 1 and 2 with 0-1, 1-2 but no 0-2 edge
    # the similarity should be between 0-1, 1-2 but 0-2 should be different from one another
    # data = create_data_naive(torch.tensor([[0], [1], [2]], dtype=torch.float), torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)).to(device)
    # data = Planetoid('./data/Planetoid', name='Cora')[0]

    # lazy load the data
    if(os.path.exists(teams_graph_input_filepath)):
        data = data_handler.load_graph(teams_graph_input_filepath)
    else:
        # this will generate graph data based on the teamsvecs.pkl file
        teamsvecs = data_handler.read_data(preprocessed_datapath)
        data = data_handler.create_graph(teamsvecs, ['member'], teams_graph_output_filepath)

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
    num_epochs = [100, 200, 500, 1000, 1500]
    # num_epochs = [100]
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
