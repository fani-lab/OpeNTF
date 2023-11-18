import pickle
import os
import numpy as np
import torch
import torch_geometric.data

import graph_params
import src.graph_params
from torch_geometric.data import Data, HeteroData
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
def create_custom_data(is_homogeneous = True):
    if(is_homogeneous):
        # create homogeneous data

        tmp = np.arange(start=0, stop=4, step=1)
        x = torch.from_numpy(tmp)
        edge_index = torch.from_numpy(np.array([[0, 1, 0, 3, 3, 4, 2, 3], \
                                                [1, 0, 3, 0, 4, 3, 3, 2]]))
        data = Data(x = x, edge_index = edge_index)
    else:
        # create heterogeneous data

        data = HeteroData()

    return data

# used to create custom teamsvecs.pkl
def create_custom_teamsvecs():
    pass


# the method to add edges of a particular edge_type to the edge_index
def add_edges(visited_index, set_edge_index, key1, key2, col1, col2, dict_key, reverse_dict_key):
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

    return set_edge_index

# generate a graph based on the sparse matrix data (e.g: teamsvecs.pkl)
# the generated graph will be saved as pickle for use in multiple models
# also this method will return the graph data as a variable
def create_homogeneous_graph(teamsvecs, node_type, output_filepath):
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

    teams_graph = Data(x = x, edge_index = edge_index)
    assert type(teams_graph) == torch_geometric.data.Data
    # save the file in the output path
    write_graph(teams_graph, output_filepath)

    return teams_graph


# create heterogeneous graph from
# teamsvecs = the information about the teams from which we have to infer the edges
# node_types = the types of nodes provided
# edge_types = the types of edges provided
def create_heterogeneous_graph(teamsvecs, node_types, edge_types, output_filepath):
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
                        set_edge_index[key2, key1][0].append(row_id)
                        set_edge_index[key2, key1][1].append(col1)
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

    # write the graph into a file for future use
    write_graph(teams_graph, output_filepath)

    # can this return type be generalized for both homo and hetero graphs?
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

    output_folder = os.path.split(output_filepath)[0]

    # at first make sure the desired folder or empty file is created
    os.makedirs(output_folder, exist_ok=True)
    with open(output_filepath, 'wb') as outputfile:
        pickle.dump(teams_graph, outputfile)
    return


if __name__ == "__main__":
    print('---------------------------------------------------------------')
    print(f'This file handles all the pickle read and write for gnn tests')
    print('---------------------------------------------------------------')

    # files and folder section
    # when used with graph_params.py
    # ----------------------------------------------------------------------
    # base_folder = graph_params.settings['storage']['base_folder']
    # output_type = graph_params.settings['storage']['output_type']
    # domain = graph_params.settings['data']['domain']
    # data_version = graph_params.settings['data']['domain']['dblp']
    # model_name = graph_params.settings['model']
    # graph_edge_type = graph_params.settings['model']
    # ----------------------------------------------------------------------

    # params when hardcoded
    params = graph_params.settings
    base_folder = params['storage']['base_folder']
    output_types = params['storage']['output_type']
    output_type = output_types[0]
    domains = list(params['data']['domain'].keys())
    domain = domains[0]
    data_versions = list(params['data']['domain'][domain].keys())
    data_version = data_versions[0]
    model_names = list(params['model'].keys())
    model_name = model_names[params['misc']['model_index']]

    node_types = params['data']['node_types']
    edge_types = params['data']['edge_types']

    graph_edge_types = list(params['model'][model_name]['edge_types'].keys())
    graph_edge_type = graph_edge_types[0]
    # file names, in the base name, the details of the versions and models will be added
    sparse_matrix_file_name = 'teamsvecs.pkl'
    graph_file_base_name = 'teamsvecs'

    ### Locations ###
    #-------------------------------------------
    teamsvecs_input_filepath = f'../../data/preprocessed/{domain}/{data_version}/{sparse_matrix_file_name}'
    # input_filepath = '../../data/preprocessed/uspt/toy.patent.tsv/teamsvecs.pkl'
    # input_filepath = '../../data/preprocessed/imdb/toy.title.basics.tsv/teamsvecs.pkl'
    # teams_graph_output_folder = f'{base_folder}/{output_type}/{domain}/{data_version}/{model_name}/{graph_edge_type}/'
    teams_graph_output_folder = f'{base_folder}/{output_type}/{domain}/{data_version}'
    # the filename should be with teamsvecs naming
    teams_graph_output_filepath = f'{teams_graph_output_folder}/{graph_file_base_name}.{model_name}.{graph_edge_type}.pkl'
    teams_graph_input_filepath = teams_graph_output_filepath
    # this is the output folder for preprocessed embeddings and performance data
    graph_preprocessed_output_folder = f'{base_folder}/preprocessed/{domain}/{data_version}'
    #-------------------------------------------

    # to make sure the path to output exists or gets created
    os.makedirs(graph_preprocessed_output_folder, exist_ok=True)
    os.makedirs(teams_graph_output_folder, exist_ok=True)

    # comment this if you dont want to test on your own data
    # x, edge_index = create_custom_data()
    # custom_teamsvecs = create_graph(x, edge_index, teams_graph_output_filepath)

    # read from a teamsvecs pickle file if you want to create graph data with that file
    teamsvecs = read_data(teamsvecs_input_filepath)

    # the graph will be made based on the mentioned node_types in graph_params
    if(len(node_types) == 1):
        create_homogeneous_graph(teamsvecs, node_types, teams_graph_output_filepath)
    else:
        create_heterogeneous_graph(teamsvecs, node_types, edge_types, teams_graph_output_filepath)
    teams_graph = load_graph(teams_graph_input_filepath)

    print(teams_graph.__dict__)