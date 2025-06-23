import os
class T2v:
    def __init__(self, output, device, cgf):
        self.name = 't2v'
        self.data = None #prepared by the _prep() like docs or a graph
        self.model = None
        self.output = output # this will be set by the children models to select which file in the output directory
        self.cfg = cgf
        self.device = device
        if not os.path.isdir(self.output): os.makedirs(self.output)
        self.modelfilepath = None

    # to prep the required dataset for embeddings like documents for d2v or graphs for gnn-based
    def _prep(self, teamsvecs, indexes, splits): pass

    def train(self, teamsvecs, indexes, splits): pass

    def get_dense_vecs(self, teamsvecs, vectype='skill'): pass # this should access the self.model to reach out the embeddings explicitly
