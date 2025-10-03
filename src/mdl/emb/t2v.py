import os
class T2v:
    def __init__(self, output, device, seed, cfg, model):
        self.data = None #prepared by the _prep() like docs or a graph
        self.name = model #str
        self.model = None #obj
        self.output = output # this will be set by the children models to select which file in the output directory
        self.cfg = cfg
        self.device = device
        self.seed = seed
        if not os.path.isdir(self.output): os.makedirs(self.output)

    # to prep the required dataset for embeddings like documents for d2v or graphs for gnn-based
    def _prep(self, teamsvecs, splits, time_indexes=None): pass

    def learn(self, teamsvecs, splits, time_indexes=None): pass

    def get_dense_vecs(self, teamsvecs, vectype='skill'): pass # this should access the self.model to reach out the embeddings explicitly
