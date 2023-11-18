settings = {
    'edge_types': {
        'ste': [('skill', '-', 'team'), ('member', '-', 'team')],
         'se': [('skill', '-', 'member')], #for any ('x', '-', 'y'), we also do ('y', '-', 'x')
          'e': 'member',
    },
    'model': {
        'max_epochs': 10,
        'embedding_dim': 5,
        'lr': 0.01,
        'batch_size': 5,
        'loader_shuffle': True,
        'num_workers': 0,
        'gcn': {
            'hidden_dim': 16,
            'dropout': 0.5,
            'p': 1.0,
            'q': 1.0,
        },
        'gat': {},
        'gin': {},
        'n2v': {
            'max_epochs': 100,
            'walk_length': 5,
            'context_size': 2,
            'walks_per_node': 10,
            'num_negative_samples': 10,
            'p' : 1.0,
            'q' : 1.0,
        },
        'm2v': {
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
        'd2v': {
            '-dm': 1, #'The training algorithm; (1: distributed memory (default), 0: CBOW')
            '-dbow_words': 0, #'Train word-vectors in skip-gram fashion; (0: no (default), 1: yes')
            '-window': 1,
            '-embtypes': 'skill', #Embedding types; (-embtypes=skill (default); member; joint; skill,member; skill,joint; ...
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