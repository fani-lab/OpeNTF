import os
class T2v:
    def __init__(self, output, device, name, cgf):
        self.data = None #prepared by the _prep() like docs or a graph
        self.name = name
        self.model = None
        self.output = output # this will be set by the children models to select which file in the output directory
        self.cfg = cgf
        self.device = device
        if not os.path.isdir(self.output): os.makedirs(self.output)

    # to prep the required dataset for embeddings like documents for d2v or graphs for gnn-based
    def _prep(self, teamsvecs, indexes, splits=None): pass #split=None >> entire dataset, no valid/test splits

    def train(self, teamsvecs, indexes, splits=None): pass #split=None >> entire dataset, no valid/test splits

    def get_dense_vecs(self, teamsvecs, vectype='skill'): pass # this should access the self.model to reach out the embeddings explicitly
