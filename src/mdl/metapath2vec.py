import graph_params
from src.misc import data_handler

import os
import torch
from torch_geometric.nn import MetaPath2Vec

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
        self.data = data_handler.load_graph(graph_datapath)

    # initialize the model
    def init(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MetaPath2Vec(self.data.edge_index_dict, embedding_dim=128,
                             metapath=self.metapath, walk_length=50, context_size=7,
                             walks_per_node=5, num_negative_samples=5,
                             sparse=True).to(self.device)

        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=1)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    # train the model to generate embeddings
    def learn(self, model, optimizer, loader, device, epoch, log_steps = 100, eval_steps = 2000):
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
        for epoch in range(num_epochs):
            self.learn(self.model, self.optimizer, self.loader, self.device, epoch)


