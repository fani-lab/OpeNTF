import pickle
import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import lil_matrix

'''
this file reads a sparse matrix (example : teamsvecs.pkl), converts it to a graph data
and also writes pickle data of custom input
'''

# reads the pickle data from the filepath
def read_data(filepath):
    try :
        with open(filepath, 'rb') as inputfile:
            teamsvecs = pickle.load(inputfile)
            # usually the teamsvecs file should contain the a dict where
            # key = ['id', 'skill', 'member'] and values will be lil_matrices
            # each lil_matrix should be inside print() to represent themselves
            # teamsvecs.todense() should show the full matrix with all elements
            print(teamsvecs.keys())
            print(teamsvecs.values())
    except FileNotFoundError:
        print('The file was not found !')
    print()
    return teamsvecs

# create your custom data here and send as x and edge_index
def create_custom_data():
    tmp = np.arange(start=1, stop=11, step=1)
    x = torch.from_numpy(tmp)
    edge_index = torch.from_numpy(np.array([[1], [1], [1], [1], [2], [3], [3], [3], [4], [4]]))

    return x, edge_index

# generate a graph based on the sparse matrix data (e.g: teamsvecs.pkl)
def create_graph_data(teamsvecs, node_type):
    # we will encounter the same node id multiple times,
    # so x as a list of nodes should be a set
    x = set()
    # we will record the visited edge indices in a dict
    # as some bool values, so that no single edge gets multiple record
    edge_index = None

    # if nodetype is 1, the graph is homogeneous
    if len(node_type) == 1:
        # generate homogeneous graph
        print(f'Generating homogeneous graph on node_type : {node_type}')

        # collect the rows and columns of the lil_matrix teamsvecs of nonzero elements only
        members = teamsvecs[node_type[0]]
        rows = members.nonzero()[0]
        cols = members.nonzero()[1]
        visited_index = {}

        # access the nonzero indices and assign an edge between the two indices
        for row_id, col_id in zip(rows, cols):
            print(f'row_id = {row_id}, col_id = {col_id}, edge_weight = {members[row_id, col_id]}')
            if(not visited_index.get((row_id, col_id), None)):
                # this row,col has not been visited

    team_graph = Data(x, edge_index)
    return team_graph

def write_data(data, filepath):
    pass

# create sample homogeneous graph data in teamsvecs.pkl file format (sparse matrix)
def create_homogeneous_graph_data(x, edge_index):
    data = Data(x = x, edge_index = edge_index)
    return data

if __name__ == "__main__":
    print(f'This file handles all the pickle read and write for gnn tests')
    # input_filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl'
    input_filepath = '../../data/preprocessed/imdb/toy.title.basics.tsv/teamsvecs.pkl'
    output_filepath = '../../data/preprocessed/custom/gnn_emb/'
    # to make sure the path to output exists or gets created
    os.makedirs(output_filepath, exist_ok=True)

    # comment this if you dont want to test on your own data
    x, edge_index = create_custom_data()
    custom_teamsvecs = create_graph_data(x, edge_index)

    # read a pickle file if you want to create graph data with that file
    teamsvecs = read_data(input_filepath)
    # the graph will be made based on the mentioned node_types
    team_graph = create_graph_data(teamsvecs, ['member'])
    print(team_graph)