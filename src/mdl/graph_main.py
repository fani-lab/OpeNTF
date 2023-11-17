from src import graph_params
from src.misc import data_handler

if __name__ == '__main__':

    params = graph_params.settings
    cmd = params['cmd']

    if 'graph' in cmd :
        # create a graph
        data_handler.create

    if 'emb' in cmd :
        # generate embeddings for the saved graph
