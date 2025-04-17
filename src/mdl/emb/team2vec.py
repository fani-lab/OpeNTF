import pickle
class Team2Vec:
    def __init__(self, dim, output, cgf):
        self.data = None #prepared by the _prep() like docs or a graph
        self.model = None
        self.output = output # this will be set by the children models to select which file in the output directory
        self.dim = dim
        self.cfg = cgf

    # to prep the required dataset for embeddings like documents for d2v or graphs for gnn-based
    def _prep(self): pass

    def train(self, epochs): pass

    def get_dense_teamsvecs(self): pass # this should access the self.model to reach out the embeddings explicitly
