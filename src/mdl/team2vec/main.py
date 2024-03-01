import argparse, pickle, os, time, sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
import torch

import params
from team2vec import Team2Vec

def addargs(parser):
    embedding = parser.add_argument_group('Team Embedding')
    embedding.add_argument('-teamsvecs', nargs='+', type=str, required=True, help='The path to the teamsvecs.pkl and indexes.pkl files; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')
    embedding.add_argument('-model', type=str, required=True, help='The embedding model; e.g., w2v, n2v, ...')
    embedding.add_argument('--output', nargs='+', type=str, required=False, help='Output folder; e.g., ../data/preprocessed/dblp/toy.dblp.v12.json/')
    embedding.add_argument('--graph_only', type=int, required=False, help='if true, then only generates graph files and returns')

def run(teamsvecs_file, indexes_file, model, output, emb_output = None):
    if not os.path.isdir(output): os.makedirs(output)
    with open(teamsvecs_file, 'rb') as teamsvecs_f, open(indexes_file, 'rb') as indexes_f:
        teamsvecs, indexes = pickle.load(teamsvecs_f), pickle.load(indexes_f)

        if model == 'w2v':
            import wnn
            settings = {'d': params.settings['model']['d'],
                        'e': params.settings['model']['e'],
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
        # this line enables to output files into folders based on different graph types
        # output_ = output + f'{params.settings["graph"]["edge_types"][1]}.{"dir" if params.settings["graph"]["dir"] else "undir"}.{str(params.settings["graph"]["dup_edge"]).lower()}/'
        output_ = output + f'{params.settings["graph"]["edge_types"][1]}.{"dir" if params.settings["graph"]["dir"] else "undir"}.{str(params.settings["graph"]["dup_edge"]).lower()}.'
        t2v = gnn.Gnn(teamsvecs, indexes, params.settings['graph'], output_)
        t2v.init() # call the team2vec's init
        if(args.graph_only):
            return

        if model == 'gnn.n2v':
            from torch_geometric.nn import Node2Vec
            t2v.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            t2v.model = Node2Vec(t2v.data.edge_index,
                                 embedding_dim=params.settings['model']['d'],
                                 walk_length=params.settings['model'][model]['walk_length'],
                                 context_size=params.settings['model'][model]['context_size'],
                                 walks_per_node=params.settings['model'][model]['walks_per_node'],
                                 num_negative_samples=params.settings['model'][model]['num_negative_samples']).to(t2v.device)

            t2v.loader = t2v.model.loader(batch_size=params.settings['model']['b'],
                                       shuffle=params.settings['model']['loader_shuffle'],
                                       num_workers=params.settings['model']['num_workers'])
            t2v.optimizer = torch.optim.Adam(list(t2v.model.parameters()), lr=params.settings['model']['lr'])
            t2v.model_name = 'n2v'

        elif model == 'gnn.m2v':
            from m2v import M2V
            from torch_geometric.nn import MetaPath2Vec

            # initialize all settings inside m2v class
            t2v = M2V(teamsvecs, indexes, params.settings, output_, emb_output)
            t2v.model_name = 'm2v'
            t2v.init() # call the m2v's init
            t2v.model = MetaPath2Vec(t2v.data.edge_index_dict, embedding_dim=t2v.settings['d'],
                                     metapath=t2v.settings['metapath'], walk_length=t2v.settings['walk_length'],
                                     context_size=t2v.settings['context_size'],
                                     walks_per_node=t2v.settings['walks_per_node'],
                                     num_negative_samples=t2v.settings['ns'],
                                     sparse=True).to(t2v.device)
            t2v.init_model()
            t2v.train(t2v.settings['e'])
            t2v.model.eval()
            emb = {}
            node_types = t2v.data._node_store_dict.keys()
            for node_type in node_types:
                emb[node_type] = t2v.model(node_type)  # output of embeddings
            embedding_output = f'{t2v.emb_output}.emb.pt'
            torch.save(emb, embedding_output, pickle_protocol=4)
            return

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
        # for edge_type in [('member', 'm')]: #n2v is only for homo, [([('skill', '-', 'team'), ('member', '-', 'team')], 'stm'), ([('skill', '-', 'member')], 'sm')]:
        for edge_type in [([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')]:
            # for dir in [True, False]:
            for dir in [False]:
                for dup in [None, 'mean']:#add', 'mean', 'min', 'max', 'mul']:
                    params.settings['graph'] = {'edge_types': edge_type, 'dir': dir, 'dup_edge': dup}
                    run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', args.model, f'{args.output}/{args.model.split(".")[0]}/', f'{args.output}/emb/')

def test_real(args):
    # test for all valid combinations on full data
    for teamsvecs in args.teamsvecs:
        args.output = teamsvecs
        # for edge_type in [([('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')]:
        for edge_type in [([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')]:
            for dir in [False]:
                for dup in ['mean']:  # add', 'mean', 'min', 'max', 'mul']:
                    params.settings['graph'] = {'edge_types': edge_type, 'dir': dir, 'dup_edge': dup}
                    run(f'{teamsvecs}teamsvecs.pkl', f'{teamsvecs}indexes.pkl', args.model,
                        f'{args.output}/{args.model.split(".")[0]}/', f'{args.output}/emb/')

# we can ignore mentioning the --output argument
#python -u main.py -teamsvecs= ./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/ -model=gnn.n2v --output=./../../../data/preprocessed/dblp/dblp.v12.json.filtered.mt75.ts3/ --graph_only 1
#python -u main.py -teamsvecs=./../../../data/preprocessed/dblp/toy.dblp.v12.json/ -model=gnn.n2v --output=./../../../data/preprocessed/dblp/toy.dblp.v12.json/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Team Embedding')
    addargs(parser)
    args = parser.parse_args()

    # run(f'{args.teamsvecs}teamsvecs.pkl', f'{args.teamsvecs}indexes.pkl', args.model, f'{args.output}/{args.model.split(".")[0]}/')

    # test_toys(args)
    test_real(args)