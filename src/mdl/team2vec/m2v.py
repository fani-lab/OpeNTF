import os.path as osp

import torch
from torch_geometric.nn import MetaPath2Vec
from gnn import Gnn
from tqdm import tqdm
from src.mdl.team2vec.params import settings

class M2V(Gnn):
    def __init__(self, teamsvecs, indexes, settings, output, emb_output): # must provide emb_output for gnn methods
        super().__init__(teamsvecs, indexes, settings, output)

        self.settings = {
            'e' : settings["model"]["e"],
            'd' : settings["model"]["d"],
            'b' : settings["model"]["b"],
            'ns' : settings["model"]["ns"],
        }
        self.model_name = 'm2v'
        self.emb_output = emb_output + f'{self.model_name}.stm.undir.mean.e{self.settings["e"]}.ns{self.settings["ns"]}.b{self.settings["b"]}.d{self.settings["d"]}' # output path of emb files

    def train(self, epoch, log_steps=100, eval_steps=2000):
        self.model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                       f'Loss: {total_loss / log_steps:.4f}'))
                total_loss = 0

            # if (i + 1) % eval_steps == 0:
            #     acc = self.test() # the method needs to be modified
            #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
            #            f'Acc: {acc:.4f}'))

    @torch.no_grad()
    def test(self, train_ratio=0.1):
        self.model.eval()

        z = self.model('member', batch=self.data['member'].y_index.to(device))
        y = self.data['member'].y

        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * train_ratio)]
        test_perm = perm[int(z.size(0) * train_ratio):]

        return self.model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                          max_iter=150)


# -teamsvecs= ./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/
# --output=./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/
if __name__ == "__main__":

    # load the graph files, or create them from the parent classes
    teamsvecs = './../../../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl'
    indexes = './../../../data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl'
    output = './../../../data/preprocessed/dblp/toy.dblp.v12.json/gnn/stm.undir.mean.'
    emb_output = './../../../data/preprocessed/dblp/toy.dblp.v12.json/emb/'


    m2v = M2V(teamsvecs, indexes, settings, output, emb_output)
    m2v.init()

    # only one metapath definition is possible
    metapath = settings["model"]["gnn.m2v"]["metapath"]
    # sample metapath from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py
    # metapath = [
    #     ('author', 'writes', 'paper'),
    #     ('paper', 'published_in', 'venue'),
    #     ('venue', 'publishes', 'paper'),
    #     ('paper', 'written_by', 'author'),
    # ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m2v.model = MetaPath2Vec(m2v.data.edge_index_dict, embedding_dim=m2v.settings["d"],
                         metapath=metapath, walk_length=2, context_size=3,
                         walks_per_node=3, num_negative_samples=m2v.settings["ns"],
                         sparse=True).to(device)

    loader = m2v.model.loader(batch_size=m2v.settings["b"], shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(m2v.model.parameters()), lr=0.01)
    for epoch in tqdm(range(1, 6)):
        m2v.train(epoch)
        # acc = m2v.test()
        # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

    emb = {}
    node_types = m2v.data._node_store_dict.keys()
    for node_type in node_types:
        emb[node_type] = m2v.model(node_type)  # output of skill embedding
    embedding_output = f'{m2v.emb_output}.emb.pt'
    torch.save(emb, embedding_output, pickle_protocol=4)