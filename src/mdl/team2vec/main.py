import argparse, pickle, os, time

import torch

import params
from team2vec import Team2Vec

def addargs(parser):
    embedding = parser.add_argument_group('Team Embedding')
    embedding.add_argument('-teamsvecs', type=str, required=True, help='The path to the teamsvecs.pkl and indexes.pkl files; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')
    embedding.add_argument('-model', type=str, required=True, help='The embedding model; e.g., w2v, n2v, ...')
    embedding.add_argument('-output', type=str, required=True, help='Output folder; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')

def run(teamsvecs_file, indexes_file, model, output):
    if not os.path.isdir(output): os.makedirs(output)
    with open(teamsvecs_file, 'rb') as teamsvecs_f, open(indexes_file, 'rb') as indexes_f:
        teamsvecs, indexes = pickle.load(teamsvecs_f), pickle.load(indexes_f)

        if model == 'w2v':
            import wnn
            settings = {'embedding_dim': params.settings['model']['embedding_dim'],
                        'max_epochs': params.settings['model']['max_epochs'],
                        'dm': params.settings['model'][model]['dm'],
                        'dbow_words': params.settings['model'][model]['dbow_words'],
                        'window': params.settings['model'][model]['dbow_words'],
                        'embtype': params.settings['model'][model]['embtype']}

            output_ = output + f'{settings["embtype"]}.'
            wnn.run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', settings, output_)
            #or
            t2v = wnn.Wnn(teamsvecs, indexes, settings, output_)
            t2v.init()
            t2v.train()
            return

        import gnn
        output_ = output + f'{params.settings["graph"]["edge_types"][1]}.{"dir" if params.settings["graph"]["dir"] else "undir"}.{str(params.settings["graph"]["dup_edge"]).lower()}/'
        t2v = gnn.Gnn(teamsvecs, indexes, params.settings['graph'], output_)
        t2v.init()
        # return

        if model == 'gnn.n2v':
            from torch_geometric.nn import Node2Vec
            t2v.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            t2v.model = Node2Vec(t2v.data.edge_index,
                                 embedding_dim=params.settings['model']['embedding_dim'],
                                 walk_length=params.settings['model'][model]['walk_length'],
                                 context_size=params.settings['model'][model]['context_size'],
                                 walks_per_node=params.settings['model'][model]['walks_per_node'],
                                 num_negative_samples=params.settings['model'][model]['num_negative_samples']).to(t2v.device)

            t2v.loader = t2v.model.loader(batch_size=params.settings['model']['batch_size'],
                                       shuffle=params.settings['model']['loader_shuffle'],
                                       num_workers=params.settings['model']['num_workers'])
            t2v.optimizer = torch.optim.Adam(list(t2v.model.parameters()), lr=params.settings['model']['lr'])
            t2v.model_name = 'n2v'

        elif model == 'gnn.m2v':
            t2v = M2V()
            t2v.model_name = 'm2v'

        # gcn (for homogeneous only)
        elif model == 'gnn.gcn':
            from gcn import Gcn as GCNModel
            t2v.model = GCNModel(hidden_channels=10, data=t2v.data)
            t2v.optimizer = torch.optim.Adam(t2v.model.parameters(), lr=params.settings['model']['lr'])
            t2v.model_name = 'gcn'

        t2v.train(params.settings['model']['max_epochs'], params.settings['model']['save_per_epoch'])
        t2v.plot_points()
        print(t2v)

def test_toys(args):
    # test for all valid combinations on toys
    for args.teamsvecs in ['./../../../data/preprocessed/dblp/toy.dblp.v12.json/',
                           './../../../data/preprocessed/imdb/toy.title.basics.tsv/',
                           './../../../data/preprocessed/gith/toy.data.csv/',
                           './../../../data/preprocessed/uspt/toy.patent.tsv/']:
        args.output = args.teamsvecs
        args.model = 'gnn.n2v'
        for edge_type in [('member', 'm')]: #n2v is only for homo, ([('skill', '-', 'team'), ('member', '-', 'team')], 'stm'), ([('skill', '-', 'member')], 'sm')]:
            for dir in [True, False]:
                for dup in [None, 'mean']:#add', 'mean', 'min', 'max', 'mul']:
                    params.settings['graph'] = {'edge_types': edge_type, 'dir': dir, 'dup_edge': dup}
                    run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', args.model, f'{args.output}/{args.model.split(".")[0]}/')

#python -u main.py -teamsvecs=./../../../data/preprocessed/dblp/toy.dblp.v12.json/ -model=gnn.n2v -output=./../../../data/preprocessed/dblp/toy.dblp.v12.json/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team Embedding')
    addargs(parser)
    args = parser.parse_args()

    # run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', args.model, f'{args.output}/{args.model.split(".")[0]}/')

    test_toys(args)
