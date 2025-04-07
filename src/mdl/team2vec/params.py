'''
During setting up the edge_types, if we want to include skill-skill or expert-expert connections,
the edge_type identifier "sm" or "stm" will have to be included as "sm.en" or "stm.en" [en = enhanced]
'''


settings = {
    'graph':{

        'edge_types':                   # this is an array holding the edge_type info in the form [('edge_type1', 'edge_type1_code'), ('edge_type2', 'edge_type2_code') ... ]
            # [('member', 'm')],
            # [([('skill', 'to', 'member')], 'sm')],
            # [([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm')], # sm enhanced
            # [([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')],
            # [([('skill', 'to', 'team'), ('member', 'to', 'team'), ('loc', 'to', 'team')], 'stml')],
            [([('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm'), ([('skill', 'to', 'team'), ('member', 'to', 'team'), ('loc', 'to', 'team')], 'stml')],  # sm, stm, stml
            # ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team'), ('skill', 'to', 'member')], 'stm') # stm enhanced,
            # [([('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm')],
            # [([('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'team'), ('member', 'to', 'team'), ('location', 'to', 'team')], 'stml')],
            # [([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm.en'), ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team'), ('skill','to','member')], 'stm.en')], # sm stm strongly connected

        'custom_supervision' : False, # if false, it will take all the forward edge_types as supervision edges
        # 'supervision_edge_types': [([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'member')], 'sm'), ([('skill', 'to', 'skill'), ('member', 'to', 'member'), ('skill', 'to', 'team'), ('member', 'to', 'team'), ('skill', 'to', 'member')], 'stm')], # sm stm strongly connected
        'supervision_edge_types': [([('skill', 'to', 'member')], 'sm.en'), ([('skill', 'to', 'team'), ('member', 'to', 'team')], 'stm.en')],

        'dir': False,
        'dup_edge': ['add', 'mean', 'min', 'max', 'mul'],         #None: keep the duplicates, else: reduce by 'add', 'mean', 'min', 'max', 'mul'
    },
    'model': {

        'd' : 8,                    # embedding dim array
        'b' : 128,                  # batch_size for loaders
        'e' : 100,                  # num epochs
        'ns' : 5,                   # number of negative samples
        'lr': 0.001,
        'loader_shuffle': True,
        'num_workers': 0,
        'save_per_epoch': False,
        'pt' : 0,                   # 1 -> use pretrained d2v skill vectors as initial node features of graph data

        'w2v': {
            'd' : 8,
            'max_epochs' : 100,
            'dm': 1,                # training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
            'dbow_words': 1,        # train word-vectors in skip-gram fashion; 0: no (default), 1: yes
            'window': 10,           # cooccurrence window
            'embtype': 'skill',     # embedding types : 'skill', 'member', 'joint', 'dt2v'
        },
        'gnn.n2v': {
            'walk_length': 5,
            'context_size': 2,
            'walks_per_node': 10,
            'num_negative_samples': 10,
            'p': 1.0,
            'q': 1.0,
        },
        'gnn.gcn': {
            'hidden_dim': 10,
            'p': 1.0,
            'q': 1.0,
        },
        'gnn.gs': {
            'e' : 100,                # number of epochs
            'b' : 128,              # batch size
            'd' : 8,                # embedding dimension
            'ns' : 5,               # number of negative samples
            'h' : 2,                # number of attention heads (if applicable)
            'nn' : [30, 20],        # number of neighbors in each hop ([20, 10] -> 20 neighbors in first hop, 10 neighbors in second hop)
            'graph_types' : 'stm',   # graph type used for a single run (stm -> ste -> skill-team-expert)
            'agg' : 'mean',         # aggregation method used for merging multiple edges between the same source and destination node
            'dir' : False,          # whether the graph is directed
        },
        'gnn.gin': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [30, 20],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
        },
        'gnn.gat': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [30, 20],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
        },
        'gnn.gatv2': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [20, 10],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
        },
        'gnn.han': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [30, 20],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
            'metapaths':{
                'sm': [[('skill', 'to', 'member'), ('member', 'rev_to', 'skill')]],
                'stm': [
                    [('member', 'to', 'team'), ('team', 'rev_to', 'skill'), ('skill', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'skill')],
                    [('member', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'skill')],
                ],
                'stml': [
                    [('member', 'to', 'team'), ('team', 'rev_to', 'loc'), ('loc', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'skill')],
                    [('member', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'skill')],
                ],
                # added one extra e-e connection in the middle
                # 'sm.en': [[('skill', 'to', 'skill'), ('skill', 'to', 'member'), ('member', 'to', 'member'), ('member', 'rev_to', 'skill'), ('skill', 'to', 'skill')]],
                'sm.en': [[('skill', 'to', 'skill'), ('skill', 'to', 'member'), ('member', 'rev_to', 'skill'), ('skill', 'to', 'skill')]],
                'stm.en': [
                    [('member', 'to', 'team'), ('team', 'rev_to', 'skill'), ('skill', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'skill')],
                    [('member', 'to', 'team'), ('team', 'rev_to', 'member')],
                    [('skill', 'to', 'team'), ('team', 'rev_to', 'skill')],
                    # repeating the same set of metapaths with additional s-s or e-e connections
                    # [('member', 'to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'skill'), ('skill', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'member')],
                    # [('skill', 'to', 'skill'), ('skill', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'skill'), ('skill', 'to', 'skill')],
                    # [('member', 'to', 'member'), ('member', 'to', 'team'), ('team', 'rev_to', 'member'), ('member', 'to', 'member')],
                    # [('skill', 'to', 'skill'), ('skill', 'to', 'team'), ('team', 'rev_to', 'skill'), ('skill', 'to', 'skill')],
                ]
            }
        },
        'gnn.gine': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [30, 20],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
        },
        'gnn.lant': {
            'e': 100,
            'b': 128,
            'd': 8,
            'ns' : 5,
            'h': 2,
            'nn': [30, 20],
            'graph_types': 'stm',
            'agg': 'mean',
            'dir': False,
        },
        'gnn.m2v': {
            'graph_types':'stm', # this value will be changed during runtime in each loop according to the graph_type and then be used in the embedding_output var
            'metapath' : {
                'sm' : [
                    ('member','rev_to','skill'),
                    ('skill', 'to', 'member'),
                ],
                'stm' : [
                    ('member','to','team'),
                    ('team', 'rev_to', 'skill'),
                    ('skill','to','team'),
                    ('team', 'rev_to', 'member'),
                ],
                'stml' : [
                    ('member','to','team'),
                    ('team', 'rev_to', 'loc'),
                    ('loc','to','team'),
                    ('team', 'rev_to', 'member'),
                ],

                # experimental section
                'sm.en' : [
                    ('member','rev_to','skill'),
                    # ('skill', 'to', 'skill'),           # additional s-s connection
                    ('skill', 'to', 'member'),
                ],
                'stm.en' : [
                    ('member','to','team'),
                    ('team', 'rev_to', 'skill'),
                    # ('skill', 'to', 'skill'),           # additional s-s connection
                    ('skill','to','team'),
                    ('team', 'rev_to', 'member'),
                ],
            },
            'walk_length': 10,
            'context_size': 10,
            'walks_per_node': 20,
            'ns' : 5,
            'd': 8,
            'b': 128,
            'e': 100,
        },
    },
    'cmd' : ['graph', 'emb'],
}