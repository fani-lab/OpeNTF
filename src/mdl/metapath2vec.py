import torch_geometric.data

import graph_params
from src.misc import data_handler

import os
import torch
from src.mdl import gnn_emb
from torch_geometric.nn import MetaPath2Vec
import numpy as np

class Metapath2Vec():

    metapath = [
        ('member','to','id'),
        ('id', 'to', 'skill'),
        ('skill','to','id'),
        ('id', 'to', 'member'),
    ]

    # setup the entire model before running
    def __init__(self):
        params = graph_params.settings
        datapath = params['misc']['graph_datapath']
        self.define_metapath(graph_params.settings['model']['metapath2vec']['metapath'])
        self.load(datapath)

    def define_metapath(self, metapath):
        self.metapath = metapath

    # this will load the desired graph data for running with the model
    def load(self, graph_datapath):
        print(f'graph data to load from : {graph_datapath}')
        self.data = data_handler.load_graph(graph_datapath)

        print(f'loaded graph data : {self.data}')

    # initialize the model
    def init(self):
        assert type(self.data) == torch_geometric.data.hetero_data.HeteroData
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MetaPath2Vec(self.data.edge_index_dict, embedding_dim=5,
                             metapath=self.metapath, walk_length=5, context_size=4,
                             walks_per_node=5, num_negative_samples=5,
                             sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=30, shuffle=True, num_workers=1)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    # train the model to generate embeddings
    def learn(self, model, optimizer, loader, device, epoch, log_steps = 2, eval_steps = 2000):
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

    def run(self, num_epochs):
        self.init()

        losses = []

        for epoch in range(num_epochs):
            self.learn(self.model, self.optimizer, self.loader, self.device, epoch)


def main(max_epochs = [10]):
    params = graph_params.settings
    m2v = Metapath2Vec()
    output_path = params['misc']['preprocessed_embedding_output_path']
    print(f'preprocessed embedding output path = {output_path}')

    for num_epochs in max_epochs:
        m2v.run(num_epochs)



if __name__ == '__main__':
    main()