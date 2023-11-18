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

class DataHandler:
    # reads the pickle data from the filepath
    # the data will not be graph
    # there is another method for loading graph data

    def __init__(self):
        self.init_variables()

    def init_variables(self):
        # params of graph_params
        self.params = graph_params.settings
        params = self.params
        # this base folder is specifically for teamsvecs sparse_matrix
        self.teamsvecs_base_folder = params['storage']['teamsvecs_base_folder']
        self.base_folder = params['storage']['base_folder']
        self.output_types = params['storage']['output_type']
        self.output_type = self.output_types[0]
        self.domains = list(params['data']['domain'].keys())
        self.domain = self.domains[0]
        self.model_names = list(params['model'].keys())
        # this model_name is set for a test purpose only
        # the graph_main.py overrides this value
        self.model_name = self.model_names[params['misc']['model_index']]

        # e.g. : ['member', 'skill']
        self.node_types = params['data']['node_types']
        # e.g. : [['skill', 'member'], ['member', 'skill']]
        self.edge_types = params['data']['edge_types']

        # 'STE', 'SE' etc
        self.graph_edge_types = list(params['model'][self.model_name]['edge_types'].keys())
        self.graph_edge_type = self.graph_edge_types[0]
        # file names, in the base name, the details of the versions and models will be added
        self.teamsvecs_file_name = 'teamsvecs.pkl'
        self.graph_file_base_name = 'teamsvecs'

        ### Locations ###
        # -------------------------------------------
        self.init_locations()
        # to make sure the path to output exists or gets created
        os.makedirs(self.graph_preprocessed_output_folder, exist_ok=True)
        os.makedirs(self.teams_graph_output_folder, exist_ok=True)

    def init_locations(self):
        params = self.params
        self.data_versions = list(params['data']['domain'][self.domain].keys())
        self.data_version = self.data_versions[0]
        self.teamsvecs_input_filepath = f'{self.teamsvecs_base_folder}/preprocessed/{self.domain}/{self.data_version}/{self.teamsvecs_file_name}'
        self.teams_graph_output_folder = f'{self.base_folder}/{self.output_type}/{self.domain}/{self.data_version}'
        # the filename should be with teamsvecs naming
        self.teams_graph_output_filepath = f'{self.teams_graph_output_folder}/{self.graph_file_base_name}.{self.graph_edge_type}.pkl'
        self.teams_graph_input_filepath = self.teams_graph_output_filepath
        # this is the output folder for preprocessed embeddings and performance data
        self.graph_preprocessed_output_folder = f'{self.base_folder}/preprocessed/{self.domain}/{self.data_version}'

    def read_data(self, filepath):
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
    def create_custom_data(self, is_homogeneous = True):
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
    def create_custom_teamsvecs(self, ):
        pass


    # the method to add edges of a particular edge_type to the edge_index
    def add_edges(self, visited_index, set_edge_index, key1, key2, col1, col2, dict_key, reverse_dict_key):
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
    def create_homogeneous_graph(self, teamsvecs, node_type, output_filepath):
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
        print(f'saving teams_graph in {output_filepath}')
        self.write_graph(teams_graph, output_filepath)

        return teams_graph


    # create heterogeneous graph from
    # teamsvecs = the information about the teams from which we have to infer the edges
    # node_types = the types of nodes provided
    # edge_types = the types of edges provided
    def create_heterogeneous_graph(self, teamsvecs, node_types, edge_types, output_filepath):
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
        print(f'saving teams_graph in {output_filepath}')
        self.write_graph(teams_graph, output_filepath)

        # can this return type be generalized for both homo and hetero graphs?
        return teams_graph

    # this particularly loads graph data from pickle file and returns the variable
    def load_graph(self, filepath):
        teams_graph = None
        try :
            with open(filepath, 'rb') as inputfile:
                teams_graph = pickle.load(inputfile)
        except FileNotFoundError:
            print(f'file {filepath} not found !')
        return teams_graph

    # write the graph data into specified pickle file
    def write_graph(self, teams_graph, output_filepath):

        output_folder = os.path.split(output_filepath)[0]

        # at first make sure the desired folder or empty file is created
        os.makedirs(output_folder, exist_ok=True)
        with open(output_filepath, 'wb') as outputfile:
            pickle.dump(teams_graph, outputfile)
        return

    # this runs and creates graph from teamsvecs
    def run(self):
        # read from a teamsvecs pickle file if you want to create graph data with that file
        teamsvecs = self.read_data(self.teamsvecs_input_filepath)

        # the graph will be made based on the mentioned node_types in graph_params
        if (len(self.node_types) == 1):
            self.create_homogeneous_graph(teamsvecs, self.node_types, self.teams_graph_output_filepath)
        else:
            self.create_heterogeneous_graph(teamsvecs, self.node_types, self.edge_types, self.teams_graph_output_filepath)

        print(f'teams_graph from DataHandler')
        teams_graph = self.load_graph(self.teams_graph_input_filepath)
        print(teams_graph.__dict__)

if __name__ == "__main__":
    print('---------------------------------------------------------------')
    print(f'This file handles all the pickle read and write for gnn tests')
    print('---------------------------------------------------------------')

    dh = DataHandler()
    dh.run()


