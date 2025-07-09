import os
import torch
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling


class FoldDataset:
    def __init__(self, all_edges, fold_indices):
        """
        all_edges: torch.Tensor of shape [2, num_edges] (src, dst)
        fold_indices: list of (train_idx, val_idx) tuples for each fold
        """
        self.all_edges = all_edges
        self.fold_indices = fold_indices

    def get_fold(self, i):
        train_idx, val_idx = self.fold_indices[i]
        return self.all_edges[:, train_idx], self.all_edges[:, val_idx]


class LinkPredictionCVRunner:
    def __init__(self, data, test_edges_ab, fold_dataset, model_fn, decoder_fn,
                 save_dir='cv_results', device='cpu', num_epochs=10, batch_size=1024):
        self.orig_data = data
        self.test_edges_ab = test_edges_ab
        self.fold_dataset = fold_dataset
        self.model_fn = model_fn
        self.decoder_fn = decoder_fn
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _to_homo_cached(self, data):
        if hasattr(data, '_homo_cache'):
            return data._homo_cache
        data._homo_cache = data.to_homogeneous()
        return data._homo_cache

    def run_fold(self, fold_idx):
        train_edges, val_edges = self.fold_dataset.get_fold(fold_idx)

        # Create fold-specific training graph
        data_fold = copy.deepcopy(self.orig_data)
        data_fold['a', 'to', 'b'].edge_index = train_edges

        homo = self._to_homo_cached(data_fold)
        ab_type_id = homo.edge_type_names.index(('a', 'to', 'b'))
        edge_type_mask = homo.edge_type == ab_type_id
        pos_edge_index = homo.edge_index[:, edge_type_mask]

        edge_dataset = TensorDataset(pos_edge_index[0], pos_edge_index[1])
        edge_loader = DataLoader(edge_dataset, batch_size=self.batch_size, shuffle=True)

        model = self.model_fn().to(self.device)
        decoder = self.decoder_fn().to(self.device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-3)

        for epoch in range(self.num_epochs):
            model.train()
            decoder.train()
            for src_pos, dst_pos in edge_loader:
                node_ids = torch.cat([src_pos, dst_pos]).unique()
                sub_loader = NeighborLoader(
                    homo,
                    input_nodes=node_ids,
                    num_neighbors=[15, 10],
                    batch_size=node_ids.size(0),
                    shuffle=False
                )
                sub_data = next(iter(sub_loader)).to(self.device)

                x = model(sub_data.x, sub_data.edge_index)
                src_pos, dst_pos = src_pos.to(self.device), dst_pos.to(self.device)
                pos_out = decoder(x[src_pos], x[dst_pos])

                neg_dst = torch.randint(0, homo.num_nodes, (len(src_pos),), device=self.device)
                neg_out = decoder(x[src_pos], x[neg_dst])

                pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
                neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
                loss = pos_loss + neg_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Save model & results
        model_path = os.path.join(self.save_dir, f'model_fold{fold_idx}.pt')
        torch.save({'model': model.state_dict(), 'decoder': decoder.state_dict()}, model_path)

        val_result = self.evaluate(model, decoder, homo, val_edges, name=f'val_fold{fold_idx}')
        test_result = self.evaluate(model, decoder, homo, self.test_edges_ab, name=f'test_fold{fold_idx}')

        result_path = os.path.join(self.save_dir, f'result_fold{fold_idx}.pt')
        torch.save({'val': val_result, 'test': test_result}, result_path)

    @torch.no_grad()
    def evaluate(self, model, decoder, homo, edge_index, name='val'):
        model.eval()
        decoder.eval()
        x = model(homo.x.to(self.device), homo.edge_index.to(self.device))

        src, dst = edge_index[0].to(self.device), edge_index[1].to(self.device)
        pos_pred = decoder(x[src], x[dst]).sigmoid()

        neg_dst = torch.randint(0, homo.num_nodes, (len(src),), device=self.device)
        neg_pred = decoder(x[src], x[neg_dst]).sigmoid()

        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        y_score = torch.cat([pos_pred, neg_pred])

        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y_true.cpu(), y_score.cpu())
        ap = average_precision_score(y_true.cpu(), y_score.cpu())
        return {'auc': auc, 'ap': ap}

    def run_all_folds(self):
        for i in range(len(self.fold_dataset.fold_indices)):
            self.run_fold(i)
        print("All folds completed.")
