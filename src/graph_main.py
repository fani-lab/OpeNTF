from src.mdl.node2vec import N2V
from src.mdl.metapath2vec import M2V
from src.misc.data_handler import DataHandler
from src import graph_params

if __name__ == '__main__':

    # all the params inside graph_params
    params = graph_params.settings
    cmd = params['cmd']

    # these params should be updated everytime in graph_main.py
    model = params['main']['model']
    domains = params['main']['domains']
    node_types = params['main']['node_types']
    # these edge_types get mapped to the actual edge type list
    # e.g. : 'STE'
    edge_types_code = params['main']['edge_types']
    # the list of edge types needed to form the graph in DataHandler
    # e.g. : [['skill', 'id'], ['id', 'skill'], ['id', 'member'], ['member', 'id']]
    edge_types = params['data']['edge_type_mapping'][edge_types_code]

    print('---------------------------------------')
    print(f'model = {model}')
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
            dh.node_types = node_types
            dh.graph_edge_type = edge_types_code
            dh.edge_types = edge_types
            dh.init_locations()
            dh.run()

        if 'emb' in cmd:
            # create the embedding
            if model == 'n2v': mobj = N2V()
            elif model == 'm2v': mobj = M2V()

            mobj.domain = domain
            mobj.graph_edge_type = edge_types_code
            mobj.init_locations()
            mobj.run()

