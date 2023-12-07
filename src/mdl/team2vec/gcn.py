import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from gnn import Gnn
from gcn_layer import Model as GCNModel
import tqdm as tqdm
from sklearn.metrics import roc_auc_score
import os
import time
import pickle

class Gcn(Gnn):
    def __init__(self, teamsvecs, indexes, settings, output):
        super().__init__(teamsvecs, indexes, settings, output)

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
            neg_sampling_ratio=1.0,
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
            edge_types = train_data.edge_types
            edge_label_index = train_data[edge_types[0]].edge_label_index
            edge_label = train_data[edge_types[0]].edge_label
            edge_label_index_tuple = (edge_types[0], edge_label_index)
        else:
            edge_label_index = train_data.edge_label_index
            edge_label = train_data.edge_label
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
            # shuffle=True,
        )

        # mini_batch_loader = HGTLoader(
        #     data = data,
        #     # Sample 20 nodes per type
        #     num_samples = [20],
        #     # Use a batch size of 128 for sampling training nodes of type paper
        #     batch_size=128,
        #     input_nodes=edge_label_index_tuple,
        # )

        # Inspect a sample:
        # sampled_data = next(iter(train_loader))
        for i, data in enumerate(mini_batch_loader):
            print(f'sample data for iteration : {i}')
            print(data)
            print(f'---------------------------------------\n')
        return mini_batch_loader

    def init_model(self, data):

        model = GCNModel(hidden_channels=10, data = data)
        print(model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: '{device}'")

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model,optimizer

    # learn for unbatched data
    def learn(self, data):
        start = time.time()
        is_directed = data.is_directed()
        min_loss = 100000000000
        epochs = 100
        emb = {}

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            data.to(device)
            pred = model(data, is_directed)

            if (type(data) == HeteroData):
                node_types = data.node_types
                edge_types = data.edge_types if is_directed else data.edge_types[
                                                                         :(len(data.edge_types)) // 2]
                # we have ground_truths per edge_label_index
                ground_truth = torch.empty(0)
                for edge_type in edge_types:
                    ground_truth = torch.cat((ground_truth, data[edge_type].edge_label.unsqueeze(0)), dim=1)
                ground_truth = ground_truth.squeeze(0)

                for node_type in node_types:
                    if(epoch == epochs):
                        emb[node_type] = model[node_type].x_dict
                # ground_truth = sampled_data['user','rates','movie'].edge_label
            else:
                if (epoch == epochs):
                    emb['node'] = model.x
                ground_truth = data.edge_label

            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()

            if(loss < min_loss):
                min_loss = loss
            if(epoch % 10 == 0):
                print(f'epoch : {epoch}, loss : {loss:.4f}')

        print(f'min_loss after {epochs} epochs : {min_loss:.4f}')
        end = time.time()
        total_time = end - start
        print(f'total time taken : {total_time:.2f} seconds || {total_time / 60:.2f} minutes || {total_time / (60 * 60)} hours')

        # store the final embeddings
        filepath2 = os.path.split(filepath)[0] + 'temp.pkl'
        with open(filepath2, 'wb') as f:
            pickle.dump(emb, f)


    # learning with batching
    def learn_batch(self, train_loader, is_directed):

        epochs = 1000

        for epoch in range(1, epochs + 1):
            total_loss = total_examples = 0
            # print(f'epoch = {epoch}')
            for sampled_data in train_loader:
                optimizer.zero_grad()

                sampled_data.to(device)
                pred = model(sampled_data, is_directed)

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
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
                # print(f'loss = {loss}')
                # print(f'epoch = {epoch}')
                # print(f'loss = {loss}')
                # print(f'total_examples = {total_examples}')
                # print(f'total_loss = {total_loss}')

            # validation part here maybe ?
            if epoch % 10 == 0 :
                # auc = eval(val_loader)
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


    # loader can be test or can be validation
    def eval(self, loader, is_directed):
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(loader):
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data, is_directed))
                # ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
                if (type(sampled_data) == HeteroData):
                    # we have ground_truths per edge_label_index
                    ground_truths.append(sampled_data[edge_type].edge_label for edge_type in sampled_data.edge_types)
                    # ground_truth = sampled_data['user','rates','movie'].edge_label
                else:
                    ground_truths = sampled_data.edge_label

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        print()
        print(f"AUC: {auc:.4f}")
        return auc

    def train(self):
        train_data, val_data, test_data = define_splits(data)

        ## Sampling
        train_loader = create_mini_batch_loader(train_data)
        # val_loader = create_mini_batch_loader(val_data)
        # test_loader = create_mini_batch_loader(test_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: '{device}'")

        # the train_data is needed to collect info about the metadata
        model, optimizer = init_model(train_data)

        # learn(train_data)
        # the sampled_data from mini_batch_loader does not properly show the
        # is_directed status
        learn_batch(train_loader, is_directed)
        # eval(test_loader)

if __name__ == '__main__':
    # homogeneous_data = create_custom_homogeneous_data()
    # heterogeneous_data = create_custom_heterogeneous_data()
    #
    # # load opentf datasets
    # filepath = '../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/m.undir.none.data.pkl'
    # data = load_data(filepath)
    # is_directed = data.is_directed()
    #
    # # # draw the graph
    # # draw_graph(data)

    from team2vec import Team2Vec
    t2v = Team2Vec()

    # train_data, val_data, test_data = define_splits(homogeneous_data)
    # train_data, val_data, test_data = define_splits(heterogeneous_data)
    train_data, val_data, test_data = define_splits(data)
    # validate_splits(train_data, val_data, test_data)

    ## Sampling
    train_loader = create_mini_batch_loader(train_data)
    # val_loader = create_mini_batch_loader(val_data)
    # test_loader = create_mini_batch_loader(test_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # the train_data is needed to collect info about the metadata
    model,optimizer = init_model(train_data)

    # learn(train_data)
    # the sampled_data from mini_batch_loader does not properly show the
    # is_directed status
    learn_batch(train_loader, is_directed)
    # eval(test_loader)