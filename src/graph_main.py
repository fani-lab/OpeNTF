from src.mdl import graph
from src.misc import data_handler
from src import graph_params

if __name__ == '__main__':

    # all the params inside graph_params
    params = graph_params.settings
    cmd = params['cmd']
    models = params['main']['models']
    domains = params['main']['domains']
    edge_types = params['main']['edge_types']

    print('---------------------------------------')
    print(f'models = {models}')
    print(f'domains = {domains}')
    print(f'edge_types = {edge_types}')
    print('---------------------------------------')

    for domain in domains:
        print(f'domain : {domain}')
        for edge in edge_types:
            print(f'edge_type : {edge_types}')
            if 'graph' in cmd:
                # create and graph

    if 'emb' in cmd:
        # create the embedding
