import pickle
import os
import numpy as np
import torch

import graph_params
import src.graph_params
from torch_geometric.data import Data
from scipy.sparse import lil_matrix

'''
this file reads a sparse matrix (example : teamsvecs.pkl), converts it to a graph data
and also writes pickle data of custom input
'''

# reads the pickle data from the filepath
# the data will not be graph
# there is another method for loading graph data
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
# the generated graph will be saved as pickle for use in multiple models
# also this method will return the graph data as a variable
def create_graph(teamsvecs, node_type, output_filepath):
    # we will encounter the same node id multiple times,
    # so x as a list of nodes should be a set
    x = set()
    # we will record the visited edge indices in a dict
    # as some bool values, so that no single edge gets multiple record
    edge_index = [[],[]]

    # if nodetype is 1, the graph is homogeneous
    if len(node_type) == 1:
        # generate homogeneous graph
        print(f'Generating homogeneous graph on node_type : {node_type}')

        # collect the rows and columns of the lil_matrix teamsvecs of nonzero elements only
        members = teamsvecs[node_type[0]]
        rows = members.nonzero()[0]
        cols = members.nonzero()[1]
        visited_index = {}
        current_row_id = -1
        col_stack = []

        # access the nonzero indices and assign an edge between the two indices
        # the members.nonzero()[0] will return the rows sorted in increasing order
        for row_id, col_id in zip(rows, cols):
            print()
            print('-----------------------------------')
            print(f'row_id = {row_id}, col_id = {col_id}')
            print('-----------------------------------')
            # adding a new member in the graph node
            # if it already exists in x, it wont be added
            x.add(col_id)
            print(f'x = {x}')
            # print(f'edge_index = {edge_index}')

            # if a new row is visited, we are traversing a new team
            # we have to reinitialize the stack of columns to create edges
            # we dont have any pair of cols to create edges this pass only
            if(current_row_id != row_id):
                print()
                print('-----------------------------------')
                print(f'starting row {row_id}')
                print(f'empty stack')
                print('-----------------------------------')
                current_row_id = row_id
                col_stack = []
            else:
                col1 = col_id
                print(f'we have to generate edge pairs for {col1} and {col_stack}')
                # iterate through each previous cols in the stack to create edges
                for i, col2 in enumerate(col_stack):

                    print(f'generate edge pair for {col1}, {col2}')
                    # if these col pair have not been visited, then create edges for these two
                    if ((not visited_index.get((col1, col2), None)) and (not visited_index.get((col2, col1), None))):
                        # mark the pair visited
                        visited_index[col1, col2] = 1
                        visited_index[col2, col1] = 1

                        # create 2 sets of edges between col1 and col2
                        edge_index[0].append(col1)
                        edge_index[1].append(col2)
                        edge_index[0].append(col2)
                        edge_index[1].append(col1)

                        print(f'generated the edge pair for {col1}, {col2}')
                        print()

                    # if they have been visited before, then you already have their edges

            col_stack.append(col_id)

    # convert the custom x and edge_index to tensors
    tmp_x = list(x)
    tmp_edge_index = np.array(edge_index)
    x = torch.tensor(list(x), dtype = torch.float64)
    edge_index = torch.tensor(np.array(edge_index), dtype = torch.long)

    teams_graph = create_homogeneous_graph_data(x, edge_index)
    # save the file in the output path
    write_graph(teams_graph, output_filepath)

    return teams_graph

# this particularly loads graph data from pickle file and returns the variable
def load_graph(filepath):
    teams_graph = None
    try :
        with open(filepath, 'rb') as inputfile:
            teams_graph = pickle.load(inputfile)
    except FileNotFoundError:
        print(f'file {filepath} not found !')
    return teams_graph

# write the graph data into specified pickle file
def write_graph(teams_graph, output_filepath):
    with open(output_filepath, 'wb') as outputfile:
        pickle.dump(teams_graph, outputfile)
    return

# create sample homogeneous graph data in teamsvecs.pkl file format (sparse matrix)
def create_homogeneous_graph_data(x, edge_index):
    data = Data(x = x, edge_index = edge_index)
    return data

if __name__ == "__main__":
    print('---------------------------------------------------------------')
    print(f'This file handles all the pickle read and write for gnn tests')
    print('---------------------------------------------------------------')

    input_filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl'
    # input_filepath = '../../data/preprocessed/uspt/toy.patent.tsv/teamsvecs.pkl'
    # input_filepath = '../../data/preprocessed/imdb/toy.title.basics.tsv/teamsvecs.pkl'
    teams_graph_output_folder = f'../../data/graph/dblp/toy.dblp.v12.json/'
    teams_graph_output_filepath = f'../../data/graph/dblp/toy.dblp.v12.json/teams_graph.pkl'
    teams_graph_input_filepath = f'../../data/graph/dblp/toy.dblp.v12.json/teams_graph.pkl'
    # this is the output folder for preprocessed embeddings and performance data
    preprocessed_output_filepath = '../../data/preprocessed/custom/gnn_emb/'
    
    # to make sure the path to output exists or gets created
    os.makedirs(preprocessed_output_filepath, exist_ok=True)
    os.makedirs(teams_graph_output_folder, exist_ok=True)

    # comment this if you dont want to test on your own data
    # x, edge_index = create_custom_data()
    # custom_teamsvecs = create_graph(x, edge_index, teams_graph_output_filepath)

    # read from a teamsvecs pickle file if you want to create graph data with that file
    teamsvecs = read_data(input_filepath)

    # the graph will be made based on the mentioned node_types
    create_graph(teamsvecs, ['member'], teams_graph_output_filepath)
    teams_graph = load_graph(teams_graph_input_filepath)

    print(teams_graph.__dict__)