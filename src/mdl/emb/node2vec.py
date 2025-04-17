# import math
#
# import torch_geometric.data
#
# import params
# import src.mdl.gnn.graph
# from src.mdl.team2vec.data_handler import DataHandler
#
# import os
# import torch
# from torch_geometric.nn import Node2Vec
# import numpy as np
#
# class N2V(src.mdl.gnn.graph.Graph):
#
#     # train the model to generate embeddings
#     def run(self):
#         losses = []; list_epochs = []; min_loss = math.inf
#         # this file logs every 10 / 20 epochs
#         # it is NOT the final pickle file for embeddings
#         with open(f'{self.graph_preprocessed_output_filename}.e{num_epochs}.txt', 'w') as outfile:
#             line = f'Graph : \n\n' \
#                    f'data = {self.data.__dict__}\n' \
#                    f'\nNumber of Epochs : {num_epochs}\n' \
#                    f'---------------------------------\n'
#             for epoch in range(num_epochs):
#                 model.train()
#                 total_loss = 0
#                 for i, (pos_rw, neg_rw) in enumerate(loader):
#                     optimizer.zero_grad()
#                     loss = model.loss(pos_rw.to(device), neg_rw.to(device))
#                     loss.backward()
#                     optimizer.step()
#
#                     print(f'\ti : {i}, loss : {loss}')
#                     total_loss += loss.item()
#                     # if (i + 1) % log_steps == 0:
#                     #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
#                     #            f'Loss: {total_loss / log_steps:.4f}'))
#                     #     total_loss = 0
#                 loss = total_loss / len(loader)
#
#                     # lines to write to file
#                     line += f'Epoch : {epoch}\n'
#                     line += f'--------------------------\n'
#                     line += f'Node ----- Embedding -----\n\n'
#                     for i, weights_per_node in enumerate(weights):
#                         print(weights_per_node)
#                         line += f'{i:2} : {weights_per_node}\n'
#                     line += f'--------------------------\n\n'
#                 losses.append(loss)
#                 list_epochs.append(epoch)
#             # write to file
#             outfile.write(line)
#
#         # store the final embeddings to a pickle file
#         self.dh.write_graph(weights, f'{self.graph_preprocessed_output_filename}.e{num_epochs}.pkl')
#
#         # draw and save the loss vs epoch plot
#         self.plot(list_epochs, losses, f'{self.graph_plot_filename}.e{num_epochs}.png')

import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora')
data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

num_workers = 4 if sys.platform == 'linux' else 0
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

# data.train_mask = data.val_mask = data.test_mask = data.y = None


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


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc


for epoch in range(1, 100):
    loss = train()
    acc = 0#test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')


@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    # plt.scatter(z[:, 0], z[:, 1], s=20)
    plt.axis('off')
    plt.show()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
plot_points(colors)