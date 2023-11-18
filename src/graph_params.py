'''
this file contains the parameters to do all the graph based tasks

1) reading teamsvecs pickle data
2) creating graph data
3) loading graph data
4) generating embeddings

'''

settings = {
    'model': {
            'gcn':{
                'edge_types' : {
                    'EE' : {
                        'graph_type' : 'homogeneous',
                    },
                    'SS' : {
                        'graph_type' : 'homogeneous',
                    },
                    'STE' : {
                        'graph_type' : 'heterogeneous',
                    },
                    'SE' : {
                        'graph_type' : 'heterogeneous',
                    },
                },
                'model_params':{
                    'max_epochs' : [10],
                    'num_features' : 1,
                    'embedding_dim' : 5,
                    'hidden_dim' : 16,
                    'dropout' : 0.5,
                    'training' : True,
                    'p' : 1.0,
                    'q' : 1.0,
                    'lr' : 0.01,
                },
            },
            'gat':{},
            'gin':{},
            'n2v':{
                'edge_types' : {
                    'EE' : {},
                    'SS' : {},
                },
                'model_params':{
                    'max_epochs' : [250],
                    'embedding_dim' : 5,
                    'walk_length' : 6,
                    'context_size' : 3,
                    'walks_per_node' : 5,
                    'num_negative_samples' : 1,
                    'p' : 1.0,
                    'q' : 1.0,
                    'lr' : 0.01,
                },
                'loader_params' : {
                    'batch_size' : 5,
                    'loader_shuffle' : True,
                    'num_workers' : 0,
                }
            },
            'm2v': {
                'edge_types' : {
                    'STE' : {},
                    'SE' : {},
                },
                'model_params':{
                    'metapath' : [
                        ('member','to','id'),
                        ('id', 'to', 'skill'),
                        ('skill','to','id'),
                        ('id', 'to', 'member'),
                    ],
                    'max_epochs' : [5],
                    'embedding_dim' : 5,
                    'walk_length' : 6,
                    'context_size' : 3,
                    'walks_per_node' : 5,
                    'num_negative_samples' : 5,
                    'batch_size' : 5,
                    'shuffle' : True,
                    'num_workers' : 1,
                    'lr' : 0.01,
                },
                'loader_params' : {
                    'batch_size' : 5,
                    'loader_shuffle' : True,
                    'num_workers' : 1
                }
            }
        },
    'data':{
        'domain': {
            'dblp':{
                'toy.dblp.v12.json':{},
            },
            'uspt':{
                'toy.patent.tsv':{},
            },
            'imdb':{
                'toy.title.basics.tsv':{},
            },
        },
        'node_types': ['member'],
        # 'node_types': ['id', 'skill', 'member'],
        'edge_types':[['skill', 'id'], ['id', 'skill'], ['id', 'member'], ['member', 'id']],
    },
    'storage' : {
        'teamsvecs_base_folder' : '../data',
        'base_folder' : '../data/graph',
        'output_type': [
            'raw',
            'preprocessed'
        ],
        'base_filename' : 'teamsvecs',
        'base_graph_emb_filename' : 'teamsvecs.emb',
        'base_graph_plot_filename' : 'teamsplot',
        'lazy_load' : True,
    },
    'cmd' : ['graph', 'emb'],
    'main':{
        'models': ['n2v','m2v'],
        'domains': ['dblp','uspt','imdb'],
        'edge_types':['STE','SE'],
    },
    'misc': {
        'graph_datapath': '../../data/graph/raw/dblp/toy.dblp.v12.json/metapath2vec/STE/teams_graph.pkl',
        'preprocessed_embedding_output_path': '../../data/graph/preprocessed/dblp/toy.dblp.v12.json/metapath2vec/STE/teamsvecs_emb.pkl',
        'domain': 'dblp',
        'dataset_version': 'toy.dblp.v12.json',
        'model': 'm2v',
        'edge_type': 'STE',
        'file_name': 'teams_graph.pkl',
        'model_index': 3,
    },
}