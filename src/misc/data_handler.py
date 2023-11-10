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

# generate a graph based on the sparse matrix data (e.g: teamsvecs.pkl)
def create_graph_data(teamsvecs : lil_matrix, node_type):
    # we will encounter the same node id multiple times,
    # so x as a list of nodes should be a set
    x = set()
    edge_index = teamsvecs.shape[1]

    # if nodetype is 1, the graph is homogeneous
    if len(node_type) == 1:
        # generate homogeneous graph
        print(f'Generating homogeneous graph on node_type : {node_type}')
        for teamsvecs[node_type]:
            pass
    team_graph =
    return team_graph

def write_data(data, filepath):
    pass

# create sample homogeneous graph data in teamsvecs.pkl file format (sparse matrix)
def create_homogeneous_graph_data(x, edge_index):
    data = Data(x = x, edge_index = edge_index)
    return data

if __name__ == "__main__":
    print(f'This file handles all the pickle read and write for gnn tests')
    input_filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl'
    output_filepath = '../../data/preprocessed/custom/gnn_emb/'
    # to make sure the path to output exists or gets created
    os.makedirs(output_filepath, exist_ok=True)

    tmp = np.arrange(start = 1, stop = 11, step = 1)
    x = torch.from_numpy(tmp)
    edge_index = torch.from_numpy(np.array([[1],[1],[1],[1],[2],[3],[3],[3],[4],[4]]))
    custom_teamsvecs = create_graph_data(x, edge_index)
    read_data(input_filepath)