import pickle
class Team2Vec:
    def __init__(self, teamsvecs, indexes, settings, output):
        self.teamsvecs = teamsvecs # key = ['id', 'skill', 'member', 'location'] and values will be lil_matrices

        self.indexes = indexes
        self.output = output
        self.settings = settings
        self.data = []
        self.model = None
        self.model_name = 't2v'

    def init(self):
        datafile = f'{self.output}data.pkl'
        try:
            print(f"Loading the data file {datafile} ...")
            with open(datafile, 'rb') as f: self.data = pickle.load(f)
            return self.data
        except FileNotFoundError:
            print(f"File not found! Generating data file {datafile} ...")
            self.create(datafile)

    def create(self, file): pass

    def train(self):pass
