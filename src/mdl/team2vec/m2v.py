import torch
from torch_geometric.nn import MetaPath2Vec
from gnn import Gnn
from tqdm import tqdm
from src.mdl.team2vec.params import settings
import os

class M2V(Gnn):
    def __init__(self, teamsvecs, indexes, settings, output, emb_output): # must provide emb_output for gnn methods
        super().__init__(teamsvecs, indexes, settings, output)

        self.model_name = 'm2v'
        self.settings = {
            'e': settings['model']['gnn.m2v']['e'],
            'd': settings['model']['gnn.m2v']['d'], # set by command line arguments
            'b': settings['model']['gnn.m2v']['b'],
            'ns': settings['model']['gnn.m2v']['ns'],
            'metapath': settings['model']['gnn.m2v']['metapath'],
            'walk_length': settings['model']['gnn.m2v']['walk_length'],
            'context_size': settings['model']['gnn.m2v']['context_size'],
            'walks_per_node': settings['model']['gnn.m2v']['walks_per_node'],
            'graph_type' : settings['model']['gnn.m2v']['graph_type']
        }
        self.emb_output = emb_output + f'{self.model_name}.{self.settings["graph_type"]}.undir.mean.e{self.settings["e"]}.ns{self.settings["ns"]}.b{self.settings["b"]}.d{self.settings["d"]}'  # output path of emb files
        if not os.path.exists(emb_output): os.makedirs(emb_output)

    def init(self):
        super().init() # create or load the graph data using team2vec's init

    # it is separated because these params are needed to set up after the model declaration
    def init_model(self):
        if self.device == 'cpu':
            self.loader = self.model.loader(batch_size=self.settings["b"], shuffle=True, num_workers=6)
        else:
            self.loader = self.model.loader(batch_size=self.settings["b"],shuffle=True)  # cuda doesnt work on the loader if num_workers param is passed
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self, num_epochs, log_steps=100, eval_steps=2000):
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.model.train()
            torch.cuda.empty_cache()

            total_loss = 0
            for i, (pos_rw, neg_rw) in enumerate(self.loader):
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if (i + 1) % log_steps == 0:
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(self.loader)}, '
                           f'Loss: {total_loss / log_steps:.4f}'))
                    total_loss = 0

                # if (i + 1) % eval_steps == 0:
                #     acc = self.test() # the method needs to be modified
                #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                #            f'Acc: {acc:.4f}'))

            # acc = m2v.test()
            # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

    @torch.no_grad()
    def test(self, train_ratio=0.1):
        self.model.eval()

        z = self.model('member', batch=self.data['member'].y_index.to(self.device))
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
    # sample metapath from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/metapath2vec.py
    # metapath = [
    #     ('author', 'writes', 'paper'),
    #     ('paper', 'published_in', 'venue'),
    #     ('venue', 'publishes', 'paper'),
    #     ('paper', 'written_by', 'author'),
    # ]

    m2v.model = MetaPath2Vec(m2v.data.edge_index_dict, embedding_dim=m2v.settings['d'],
                         metapath=m2v.settings['metapath'], walk_length=m2v.settings['walk_length'], context_size=m2v.settings['context_size'],
                         walks_per_node=m2v.settings['walks_per_node'], num_negative_samples=m2v.settings['ns'],
                         sparse=True).to(m2v.device)

    m2v.init()
    m2v.train(m2v.settings['e'])
    m2v.model.eval()
    emb = {}
    node_types = m2v.data._node_store_dict.keys()
    for node_type in node_types:
        emb[node_type] = m2v.model(node_type)  # output of skill embedding
    embedding_output = f'{m2v.emb_output}.emb.pt'
    torch.save(emb, embedding_output, pickle_protocol=4)