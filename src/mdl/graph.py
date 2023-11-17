
import torch
import torch_geometric.data
import graph_params
from src.misc import data_handler
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

import math
import os

class Graph:

    def __init__(self):
        self.init_variables()

    def init_variables(self):

        # detect the model name, then access the parameters specific to that model
        if('GNN' in str(self.__class__)):
            print(f'Model name is gnn')
            self.model_name = 'gnn'
        elif('GCN' in str(self.__class__)):
            print(f'Model name is gcn')
            self.model_name = 'gcn'
        elif ('GAT' in str(self.__class__)):
            print(f'Model name is gat')
            self.model_name = 'gat'
        elif ('GIN' in str(self.__class__)):
            print(f'Model name is gin')
            self.model_name = 'gin'
        elif ('N2V' in str(self.__class__)):
            print(f'Model name is n2v')
            self.model_name = 'n2v'
        elif ('M2V' in str(self.__class__)):
            print(f'Model name is m2v')
            self.model_name = 'm2v'


        self.params = graph_params.settings
        params = self.params

        # model parameters
        model_params = params['model'][self.model_name]['model_params']
        self.embedding_dim = model_params['embedding_dim']
        self.max_epochs = model_params['max_epochs']
        self.walk_length = model_params['walk_length']
        self.walks_per_node = model_params['walks_per_node']
        self.context_size = model_params['context_size']
        self.num_negative_samples = model_params['num_negative_samples']

        # loader parameters
        loader_params = params['model'][self.model_name]['loader_params']
        self.num_workers = loader_params['num_workers']
        self.batch_size = loader_params['batch_size']
        self.loader_shuffle = loader_params['loader_shuffle']

        self.output_types = params['storage']['output_type']
        self.output_type = self.output_types[0]
        self.domains = list(params['data']['domain'].keys())
        self.domain = self.domains[0]
        self.data_versions = list(params['data']['domain'][self.domain].keys())
        self.data_version = self.data_versions[0]
        self.model_names = list(params['model'].keys())
        # only for n2v, the edge_types will be E-E, S-S or T-T
        self.graph_edge_types = list(params['model'][self.model_name]['edge_types'].keys())
        self.graph_edge_type = self.graph_edge_types[0]

        ### Locations ###
        # -------------------------------------------
        # teams_graph_output_folder = f'{base_folder}/{output_type}/{domain}/{data_version}/{model_name}/{graph_edge_type}/'
        # base_filename
        # base_graph_emb_filename
        # base_graph_plot_filename
        self.lazy_load = params['storage']['lazy_load']
        self.base_folder = params['storage']['base_folder']
        self.base_filename = params['storage']['base_filename']
        self.teams_graph_output_folder = f'{self.base_folder}/{self.output_type}/{self.domain}/{self.data_version}'
        self.teams_graph_output_filepath = f'{self.teams_graph_output_folder}/{self.base_filename}.{self.model_name}.{self.graph_edge_type}.pkl'
        self.teams_graph_input_filepath = self.teams_graph_output_filepath

        # this is the output folder for preprocessed embeddings and performance data
        self.base_graph_emb_filename = params['storage']['base_graph_emb_filename']
        self.graph_preprocessed_output_folder = f'{self.base_folder}/preprocessed/{self.domain}/{self.data_version}'
        # need to add '{epoch_number}.pkl' while running at each set of epochs
        self.graph_preprocessed_output_filename = f'{self.graph_preprocessed_output_folder}/{self.base_graph_emb_filename}.{self.model_name}.{self.graph_edge_type}'

        # need to add '{epoch_number}.png' while running at each set of epochs
        self.base_graph_plot_filename = params['storage']['base_graph_plot_filename']
        self.graph_plot_filename = f'{self.graph_preprocessed_output_folder}/{self.base_graph_plot_filename}.{self.model_name}.{self.graph_edge_type}.'
        # -------------------------------------------

    def init_model(self):
        pass

    def learn(self, model, optimizer, loader, device, epoch, log_steps=50, eval_steps=2000):
        pass

    def load(self, graph_datapath):
        pass

    # normalizr an np_array into a range of 0-1
    def normalize(self, np_array):
        mx = np.max(np_array)
        mn = np.min(np_array)

        # print(f'\nNormalize()\n')
        # print(f'Before :')
        # print(f'{np_array}')

        np_array = (np_array - mn) / (mx - mn)

        # print(f'After :')
        # print(f'{np_array}')
        # print('')

        return np_array

    # to plot a graph performance
    def plot(self, x, y, output_filepath = None):
        mx = max(y)
        mn = min(y)
        threshold = (mx - mn) // 10

        plt.figure()
        plt.ylabel('Loss')
        plt.ylim(mn - threshold, mx + threshold)
        plt.xlabel('Epochs')
        plt.xlim(0, len(x))
        plt.plot(x, y)
        plt.legend('This is a legend')
        # save the figure before showing the plot
        # because after plt.show() another blank figure is created
        if(output_filepath):
            print(f'Saving plot in {output_filepath}')
            plt.savefig(output_filepath)

        plt.show()

    # for plotting data for multiple sets of epochs or something similar
    def multi_plot(self):
        pass

    def plot_homogeneous_graph(self, teams_graph):
        # make sure that the data given is a Data type object
        assert type(teams_graph) == torch_geometric.data.Data

        G = torch_geometric.utils.to_networkx(teams_graph, to_undirected=True)
        nx.draw(G, with_labels=True)

    def plot_heterogeneous_graph(self, teams_graph):
        # make sure that the data given is a Data type object
        assert type(teams_graph) == torch_geometric.data.hetero_data.HeteroData
        pass