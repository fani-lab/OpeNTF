import pickle
import os
import numpy as np
import torch
from scipy.sparse import lil_matrix

# reads the pickle data from the filepath
def read_data(filepath):
    try :
        with open(filepath, 'rb') as inputfile:
            teamsvecs = pickle.load(inputfile)
            # usually the teamsvecs file should contain the a dict where
            # key = ['id', 'skill', 'member'] and values will be lil_matrices
            # each lil_matrix should be inside print() to represent themselves
            print(teamsvecs.keys())
            print(teamsvecs.values())
    except FileNotFoundError:
        print('The file was not found !')
    print()
    return teamsvecs

def write_data(data, filepath):
    pass

# create sample graph data in teamsvecs.pkl file format (sparse matrix)
def create_graph_data(x, edge_index):
    data = None
    return data

if __name__ == "__main__":
    print(f'This file handles all the pickle read and write for gnn tests')
    input_filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl'
    output_filepath = '../../data/preprocessed/custom/gnn_emb/'
    # to make sure the path to output exists or gets created
    os.makedirs(output_filepath, exist_ok=True)

    tmp = np.arrange(start = 1, stop = 11, step = 1)
    x = torch.from_numpy([[1],[1],[1],[1],[2],[3],[3],[3],[4],[4],])
    edge_index = torch.from_numpy()
    custom_teamsvecs = create_graph_data(x, edge_index)
    read_data(input_filepath)