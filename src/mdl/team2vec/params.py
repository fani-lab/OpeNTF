settings = {
    'graph':{
        'edge_types':
            ('member', 'm'),
            # ([('skill', '-', 'team'), ('member', '-', 'team')], 'stm'),
            # ([('skill', '-', 'member')], 'sm'),
        'dir': False,
        'dup_edge': None, #None: keep the duplicates, else: reduce by 'add', 'mean', 'min', 'max', 'mul'
    },
    'model': {
        'max_epochs': 100,
        'embedding_dim': 5,
        'lr': 0.01,
        'batch_size': 5,
        'loader_shuffle': True,
        'num_workers': 0,
        'save_per_epoch': False,
        'w2v': {
            'dm': 1,  # training algorithm. 1: distributed memory (PV-DM), 0: distributed bag of words (PV-DBOW)
            'dbow_words': 0,  # 'train word-vectors in skip-gram fashion; 0: no (default), 1: yes
            'window': 2,  # cooccurrence window
            'embtype': 'joint',  # 'member', 'joint', 'dt2v'
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
        'gnn.gat': {},
        'gnn.gin': {},
        'gnn.m2v': {
            'metapath' : [
                ('member','to','id'),
                ('id', 'to', 'skill'),
                ('skill','to','id'),
                ('id', 'to', 'member'),
            ],
            'walk_length': 5,
            'context_size': 3,
            'walks_per_node': 10,
            'num_negative_samples' : 10,
        },
    },
    'data':{
        'dblp':{},
        'uspt':{},
        'imdb':{},
        'node_types': ['member'], #['id', 'skill', 'member'],
    },
    'cmd' : ['graph', 'emb'],
    'main':{
        'model': 'm2v',
        'domains': ['uspt','imdb','dblp'],
        'node_types': ['id', 'skill', 'member'],
        'edge_types': 'STE',
    },
}