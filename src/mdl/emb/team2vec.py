import pickle
class Team2Vec:
    def __init__(self, dim, output, device, cgf):
        self.data = None #prepared by the _prep() like docs or a graph
        self.model = None
        self.output = output # this will be set by the children models to select which file in the output directory
        self.dim = dim
        self.cfg = cgf
        self.device = device

    # to prep the required dataset for embeddings like documents for d2v or graphs for gnn-based
    def _prep(self, teamsvecs, indexes): pass

    def train(self, epochs, teamsvecs, indexes): pass

    def get_dense_vecs(self): pass # this should access the self.model to reach out the embeddings explicitly
