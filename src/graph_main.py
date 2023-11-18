from src.mdl import node2vec, metapath2vec
from src.misc.data_handler import DataHandler
from src import graph_params

if __name__ == '__main__':

    # all the params inside graph_params
    params = graph_params.settings
    cmd = params['cmd']
    # these are the params for graph_main.py
    models = params['main']['models']
    domains = params['main']['domains']
    edge_types = params['main']['edge_types']

    print('---------------------------------------')
    print(f'models = {models}')
    print(f'domains = {domains}')
    print(f'edge_types = {edge_types}')
    print('---------------------------------------')

    # instantiate the objects
    dh = DataHandler()

    for domain in domains:
        print()
        ('---------------------------------------')
        ('---------------------------------------')
        print(f'domain : {domain}')
        ('---------------------------------------')
        # change edge of the classes

        if 'graph' in cmd:
            # create the graph
            dh.domain = domain
            dh.run()

        # if 'emb' in cmd:
            # create the embedding
            # n2v = node2vec.N2V()
            # m2v = metapath2vec.M2V()
