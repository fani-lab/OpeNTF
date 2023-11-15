'''
this file contains the parameters to do all the graph based tasks

1) reading teamsvecs pickle data
2) creating graph data
3) loading graph data
4) generating embeddings

'''

settings = {
    'model':{
            'gnn':{},
            'gcn':{},
            'gan':{},
            'gin':{},
            'node2vec':{},
            'metapath2vec':{
                'metapath' : [
                    ('member','to','id'),
                    ('id', 'to', 'skill'),
                    ('skill','to','id'),
                    ('id', 'to', 'member'),
                ],
                'STE' : {},
                'SE' : {},
                'STE_TL' : {},
                'STEL' : {}
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
    },
    'storage':{
        'base_folder' : '../../data/graph/',
        'output_type': [
            'raw',
            'preprocessed'
        ],
    },
    'misc':{
        'graph_datapath' : '../../data/graph/raw/dblp/toy.dblp.v12.json/metapath2vec/STE/teams_graph.pkl',
        'preprocessed_embedding_output_path' : '../../data/graph/preprocessed/dblp/toy.dblp.v12.json/metapath2vec/STE/teamsvecs_emb.pkl',
        'domain' : 'dblp',
        'dataset_version' : 'toy.dblp.v12.json',
        'model' : 'metapath2vec',
        'edge_type' : 'STE',
        'file_name' : 'teams_graph.pkl',
    }
}