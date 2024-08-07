'''
We create a separate encoder class for lant due to the difference
in training and loss calculation mechanism in lants
'''

import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import to_hetero
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch import nn as nn
from torch_geometric.nn import DeepGraphInfomax as DGI, GATConv
from lant import LANT, calculate_loss, calculate_link_prediction_loss_auc, summary_function, corruption
from mdl.earlystopping import EarlyStopping
from decoder import Decoder
import params

'''
This is functionally a bit different from the Encoder class of encoder.py
'''
class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.heads = 2

        '''
        this section of linear transformation is needed to add random feature matrix to the initial nodes with 1 dimensional features
        (as our data doesnt have its own node features. But the intricate part here is that we at first linear transform the node features
        into our desired dimensions = hidden_channels. But then before feeding to the dgi encoder, we have to divide it by the number of 
        attention heads of the GATConv layer. Because after feeding into the GAT encoder of the dgi, the output channels will multiply by 
        the number of heads and so the initial dimension would be increased. To counter that, we do the early division by heads before 
        feeding to the GAT encoder)
        '''
        if (type(data) == HeteroData):
            self.node_lin = []
            self.node_emb = []
            # for each node_type
            node_types = data.node_types
            # linear transformers and node embeddings based on the num_features and num_nodes of the node_types
            # these two are generated such that both of them has the same shape and they can be added together
            for i, node_type in enumerate(node_types):
                if (data.is_cuda):
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels).cuda())
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels).cuda())
                else:
                    self.node_lin.append(nn.Linear(data[node_type].num_features, hidden_channels))
                    self.node_emb.append(nn.Embedding(data[node_type].num_nodes, hidden_channels))

        self.hidden_channels = hidden_channels // self.heads # the attention heads cause the dimension to double inside the model

        self.encoder = to_hetero(LANT(self.hidden_channels, self.hidden_channels, heads = self.heads), metadata=data.metadata())
        # self.encoder = to_hetero(LANTbk(hidden_channels, hidden_channels), metadata=data.metadata())

        self.dgi = DGI(
            hidden_channels=self.hidden_channels * self.heads,
            encoder=self.encoder,
            summary=summary_function,
            corruption=corruption
        )

        # self.dgi = to_hetero(self.dgi, metadata=data.metadata())

    def forward(self, data, seed_edge_type, is_directed, emb=False) -> Tensor:
        if (type(data) == HeteroData):
            self.x_dict = {}
            for i, node_type in enumerate(data.node_types):
                self.x_dict[node_type] = self.node_lin[i](data[node_type].x) + self.node_emb[i](data[node_type].n_id)

        pos_z, neg_z, summary = self.dgi(self.x_dict, data.edge_index_dict)

        if emb: return pos_z

        return pos_z, neg_z, summary


    # the gnn object can provide the class variables
    def learn(self, gnn, epochs):
        self.device = gnn.device
        start = time.time()
        train_loader_dict = gnn.train_loader
        val_loader_dict = gnn.val_loader
        epochs_taken = 0

        pos_z = {}
        loss_array = []
        val_auc_array = []
        val_loss_array = []
        earlystopping = EarlyStopping(patience=5, verbose=True, delta=0.001,
                                      path=f"{gnn.model_output}/state_dict_model.e{epochs}.pt", trace_func=print,
                                      save_model=False)
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for seed_edge_type in params.settings['graph']['supervision_edge_types']:
                for sampled_data in train_loader_dict[seed_edge_type]:
                    sampled_data.to(self.device)
                    gnn.optimizer.zero_grad()
                    pos_z, neg_z, summary = self(sampled_data, seed_edge_type, gnn.is_directed) # the forward of this own model
                    # print(f"pos_z (team): {pos_z['team']}")
                    # print(f"neg_z (team): {neg_z['team']}")

                    # Mutual Information Loss
                    mi_loss = calculate_loss(gnn.model.dgi, pos_z, neg_z, summary)

                    # Link Prediction Loss and auc for training data (auc is ignored here)
                    # lp_loss, _ = calculate_link_prediction_loss_auc(pos_z, sampled_data, seed_edge_type)

                    # Total Loss
                    total_loss = mi_loss

                    total_loss.backward()
                    gnn.optimizer.step()

                    epoch_losses.append(total_loss.item())

            # Average losses for the epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

            loss_array.append(avg_epoch_loss)

            print(f'e : {epoch}, l : {avg_epoch_loss}')

            gnn.optimizer.zero_grad()
            ls = self.eval(gnn, val_loader_dict)
            val_loss_array.append(ls)

            epochs_taken += 1
            earlystopping(val_loss_array[-1], self)
            if earlystopping.early_stop:
                print(f"Early Stopping Triggered at epoch: {epoch}")
                break

        # plot the figure and save
        fig_output = f'{gnn.model_output}/{gnn.model_name}.{gnn.graph_type}.undir.{gnn.agg}.e{epochs}.ns{int(gnn.ns)}.b{gnn.b}.d{gnn.d}.png'
        gnn.plot_graph(torch.arange(1, epochs_taken + 1, 1), loss_array, val_loss_array, fig_output=fig_output)
        # fig_output = f'{gnn.model_output}/{gnn.model_name}.{gnn.graph_type}.undir.{gnn.agg}.e{epochs}.ns{int(gnn.ns)}.b{gnn.b}.d{gnn.d}.val_auc_per_epoch.png'
        # gnn.plot_graph(torch.arange(1, epochs_taken + 1, 1), val_auc_array, xlabel='Epochs', ylabel='Val AUC',
        #                 title=f'Validation AUC vs Epochs for Embedding Generation', fig_output=fig_output)
        print(f'\nit took {(time.time() - start) / 60} mins || {(time.time() - start) / 3600} hours to train the model\n')


        torch.cuda.empty_cache()

    @torch.no_grad
    def eval(self, gnn, val_loader_dict):
        total_loss = 0.0
        num_batches = 0

        # Iterate over each edge type in the validation loader dictionary
        for seed_edge_type, val_loader in val_loader_dict.items():
            for sampled_data in val_loader:
                sampled_data = sampled_data.to(self.device)

                # Forward pass to get pos_z, neg_z, and summary
                pos_z, neg_z, summary = self(sampled_data, seed_edge_type, gnn.is_directed)

                # Calculate the mutual information loss
                mi_loss = calculate_loss(gnn.model.dgi, pos_z, neg_z, summary)

                total_loss += mi_loss.item()
                num_batches += 1

        # Calculate the average loss
        average_loss = total_loss / num_batches

        print(f'Validation Loss: {average_loss}')

        return average_loss

    # this eval performs validation incorporating link prediction
    # index out of bounds issue occurs while performing dot product
    @torch.no_grad
    def eval_with_lp_loss(self, pos_z_dict, val_loader_dict):
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        # Iterate over each edge type in the validation loader dictionary
        for seed_edge_type, val_loader in val_loader_dict.items():
            for sampled_data in val_loader:
                sampled_data.to(self.device)
                # Get the edge_label_index and edge_label for the current batch and current seed_edge_type
                edge_label_index = sampled_data[seed_edge_type].edge_label_index
                edge_label = sampled_data[seed_edge_type].edge_label

                preds = []
                for i in range(edge_label_index.shape[1]):
                    src, dst = edge_label_index[:, i]
                    pred = (pos_z_dict[seed_edge_type[0]][src] * pos_z_dict[seed_edge_type[2]][dst]).sum(dim = -1).unsqueeze(0)
                    preds.append(pred)

                preds = torch.cat(preds)

                # Calculate the binary cross entropy loss
                lp_loss = F.binary_cross_entropy_with_logits(preds, edge_label.float())
                total_loss += lp_loss.item()

                # Collect the predictions and labels for AUC calculation
                all_preds.append(preds.detach().cpu())
                all_labels.append(edge_label.detach().cpu())

                num_batches += 1

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate the AUC score
        auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

        # Calculate the average loss
        average_loss = total_loss / num_batches

        print(f'val loss : {average_loss}, val auc : {auc}')

        return average_loss, auc