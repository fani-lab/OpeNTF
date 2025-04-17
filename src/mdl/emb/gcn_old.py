import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from gnn import Gnn
from gcn_layer_old import Model as GCNModel
import tqdm as tqdm
from sklearn.metrics import roc_auc_score
import os, argparse, pickle, time

class Gcn(Gnn):
    def __init__(self, teamsvecs, indexes, settings, output): super().__init__(teamsvecs, indexes, settings, output)

    def define_splits(self, data):

        if(type(data) == HeteroData):
            num_edge_types = len(data.edge_types)

            # directed graph means we dont have any reverse edges
            if(data.is_directed()):
                edge_types = data.edge_types
                rev_edge_types = None
            else :
                edge_types = data.edge_types[:num_edge_types // 2]
                rev_edge_types = data.edge_types[num_edge_types // 2:]
        else:
            edge_types = None
            rev_edge_types = None


        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            neg_sampling_ratio=0.0,
            add_negative_train_samples=False,
            edge_types= edge_types,
            rev_edge_types=rev_edge_types,
        )

        train_data, val_data, test_data = transform(data)

        return train_data, val_data, test_data

    def create_mini_batch_loader(self, data):
        # Define seed edges:
        # we pick only a single edge_type to feed edge_label_index (need to verify this approach)
        if (type(data) == HeteroData):
            edge_types = data.edge_types
            edge_label_index = data[edge_types[0]].edge_label_index
            edge_label = data[edge_types[0]].edge_label
            edge_label_index_tuple = (edge_types[0], edge_label_index)
        else:
            edge_label_index = data.edge_label_index
            edge_label = data.edge_label
            edge_label_index_tuple = edge_label_index
        print(f'edge_label stuffs : {edge_label_index}, {edge_label}')

        mini_batch_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=0.0,
            edge_label_index = edge_label_index_tuple,
            # edge_label_index = None,
            edge_label = edge_label,
            # edge_label = None,
            batch_size=32,
            shuffle=True,
        )

        return mini_batch_loader

    def learn_batch(self, epochs, train_loader, is_directed):
        for epoch in range(epochs):
            total_loss = total_examples = 0
            # print(f'epoch = {epoch}')
            for sampled_data in train_loader:
                self.optimizer.zero_grad()

                sampled_data.to(self.device)
                pred = self.model(sampled_data, is_directed)

                # The ground_truth and the pred shapes should be 1-dimensional
                # we squeeze them after generation
                if(type(sampled_data) == HeteroData):
                    edge_types = sampled_data.edge_types if is_directed else sampled_data.edge_types[:(len(sampled_data.edge_types)) // 2]
                    # we have ground_truths per edge_label_index
                    ground_truth = torch.empty(0)
                    for edge_type in edge_types:
                        ground_truth = torch.cat((ground_truth, sampled_data[edge_type].edge_label.unsqueeze(0)), dim = 1)
                    ground_truth = ground_truth.squeeze(0)
                    # ground_truth = sampled_data['user','rates','movie'].edge_label
                else:
                    ground_truth = sampled_data.edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

            # validation part here maybe ?
            if epoch % 10 == 0 :
                # auc = eval(val_loader)
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    def eval(self, data, mode='validation'):
        with torch.no_grad():
            data.to(self.device)
            pred = self.model(data, self.is_directed)
            # The ground_truth and the pred shapes should be 1-dimensional
            # we squeeze them after generation
            if (type(data) == HeteroData):
                edge_types = data.edge_types if self.is_directed else data.edge_types[:(len(data.edge_types)) // 2]
                # we have ground_truths per edge_label_index
                ground_truth = torch.empty(0)
                for edge_type in edge_types: ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
                ground_truth = ground_truth.squeeze(0)
            else:
                ground_truth = data.edge_label

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

        print(f'{mode} loss : {loss:.4f}')
        print(f'pred = {pred}')
        print(f'ground_truth = {ground_truth}')
        print(f'pred = {pred.sigmoid()}')

        if (mode == 'validation'):
            auc = roc_auc_score(ground_truth, torch.sigmoid(pred))
            print()
            print(f"AUC : {auc:.4f}")
            return auc

    def train(self, epochs, save_per_epoch=False):
        self.is_directed = self.data.is_directed()
        self.train_data, self.val_data, self.test_data = self.define_splits(self.data)

        ## Sampling
        # train_loader = self.create_mini_batch_loader(self.train_data)
        # val_loader = create_mini_batch_loader(val_data)
        # test_loader = create_mini_batch_loader(test_data)

        # the sampled_data from mini_batch_loader does not properly show the
        # is_directed status
        # self.learn_batch(epochs, train_loader, is_directed)
        # eval(test_loader)
        t_start_time = time.time()
        min_loss = 100000000000
        emb = {}

        for epoch in range(epochs):
            e_start_time = time.time()
            self.optimizer.zero_grad()
            data.to(self.device)
            pred = self.model(data, self.is_directed)

            if (type(data) == HeteroData):
                node_types = data.node_types
                edge_types = data.edge_types if self.is_directed else data.edge_types[:(len(data.edge_types)) // 2]
                # we have ground_truths per edge_label_index
                ground_truth = torch.empty(0)
                for edge_type in edge_types: ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
                ground_truth = ground_truth.squeeze(0)

                for node_type in node_types:
                    if(epoch == epochs): emb[node_type] = self.model[node_type].x_dict
                # ground_truth = sampled_data['user','rates','movie'].edge_label
            else:
                if (epoch == epochs): emb['node'] = self.model.x
                ground_truth = data.edge_label

            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            self.optimizer.step()

            if(loss < min_loss): min_loss = loss
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Time: {time.time() - e_start_time: 0.4f}')
            if save_per_epoch: torch.save(self.model.state_dict(), f'{model_output}/gnn_model.e{epoch}.pt', pickle_protocol=4)
        torch.save(self.model.state_dict(), f'{model_output}/gnn_model.pt', pickle_protocol=4)
        # to load later by: self.model.load_state_dict(torch.load(f'{self.output}/gnn_model.pt'))

        print(f'It took {time.time() - t_start_time} to train the model.')
        # store the final embeddings
        filepath = self.output
        # filepath2 = os.path.split(filepath)[0] + 'temp.pkl'
        # with open(filepath2, 'wb') as f:
        #     pickle.dump(emb, f)
