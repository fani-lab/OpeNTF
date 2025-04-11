import gensim, pickle
import getopt, sys, multiprocessing

import dal.load_dblp_data as dblp


class Embedding:
    """A class used to generate embeddings and dictionaries
    Creates skills and experts embeddings in order for the VAE to
    easily process the dataset
     ----------
     database_name : string
         The user-provided dataset name that the VAE architecture
         will be applied on
     database_path : string
         The user-provided dataset path that the VAE architecture
         will be applied on
     embeddings_save_path : string, optional (default='output/Models/T2V/')
         The user-provided/default path where the embedded verions of
         the dataset will be stored
     """
    def __init__(self, database_name, database_path, embeddings_save_path='output/Models/dblp/'):
        self.database_name = database_name
        self.databasePath = database_path
        self.embeddings_save_path = embeddings_save_path
        self.teams = []
        self.member_type = ''

    def get_database_name(self):
        """Returns the database name provided by the user
        """
        return self.database_name

    def get_database_path(self):
        """Returns the database path provided by the user
        """
        return self.databasePath

    def embeddings_save_path(self):
        """Returns the embedding save path provided by the user
        """
        return self.embeddings_save_path

    def init(self, team_matrix, member_type='skill'):  # member_type={'user','skill'}
        """Initialize embedding model
        Extract user/skills indices from the database to create an embedding
        model that wll be used for training
        Parameters
        ----------
        team_matrix : array-like, shape=(n_teams,)
            The team_matrix array. The combination of users and skills that
            makes up each team in the database
        member_type : string
            The type of embedding to be computed ('user' or 'skill')
        """
        self.member_type = member_type
        teams_label = []
        # teams_skills = []
        teams_members = []
        for team in team_matrix:  # iterate through ever team in the team matrix
            teams_label.append(team[0])
            if member_type.lower() == 'skill':
                teams_members.append(team[1].col)
            else:  # member_type == 'user'
                teams_members.append(team[2].col)

        for index, team in enumerate(teams_members):
            # apply the doc2vec model for every skill in each collaboration
            td = gensim.models.doc2vec.TaggedDocument([str(m) for m in team], [
                str(teams_label[index])])  # the [] is needed to surround the tags!
            self.teams.append(td)
        print('#teams loaded: {}; member type = {}'.format(len(self.teams), member_type))

    def train(self, dimension=300, window=2, dist_mode=1, epochs=100, output=embeddings_save_path):
        """Train the teams that were previously generated
        Train user/skill teams and save in vector format
        Parameters
        ----------
        dimension : int
            The dimensionality of the word vectors
        window : int
            The maximum distance between the current and predicted word
            within a sentence.
        dist_mode : int
            Whether to choose distributed memory or bag of words for the
            embedding
        epochs : int
            The number of iteration over the corpus
        output : string
            The local path where the embeddings will be saved
        """
        self.settings = 'd' + str(dimension) + '_w' + str(window) + '_m' + str(dist_mode) + '_t' + str(self.member_type.capitalize())
        print('training settings: %s\n' % self.settings)

        # build the model
        # alpha=0.025
        # min_count=5
        # max_vocab_size=None
        # sample=0
        # seed=1
        # min_alpha=0.0001
        # hs=1
        # negative=0
        # dm_mean=0
        # dm_concat=0
        # dm_tag_count=1
        # docvecs=None
        # docvecs_mapfile=None
        # comment=None
        # trim_rule=None

        # apply the doc2vec model on the settings
        self.model = gensim.models.Doc2Vec(dm=dist_mode,
                                           # ({1,0}, optional)  Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
                                           vector_size=dimension,
                                           window=window,
                                           dbow_words=1,
                                           # ({1,0}, optional)  If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
                                           min_alpha=0.025,
                                           min_count=0,
                                           workers=multiprocessing.cpu_count())
        self.model.build_vocab(self.teams)  # build vocabulary for the teams

        # start training
        for e in range(epochs):
            if not (e % 10):
                print('iteration {0}'.format(e))
            self.model.train(self.teams, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.model.alpha -= 0.002  # decrease the learning rate
            self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay

        # save the model
        if output:
            with open('{}teams_{}'.format(output, self.settings), 'wb') as f:
                pickle.dump(self.teams, f)
            self.model.save('{}model_{}'.format(output, self.settings))
            self.model.save_word2vec_format('{}members2vec_{}'.format(output, self.settings))
            self.model.docvecs.save_word2vec_format('{}team2vec_{}'.format(output, self.settings))
            print('Model saved for {} under directory {}'.format(self.settings, output))

    def get_team_vec(self, tid):
        """Returns the team data in vector format
        Parameters
        ----------
        tid : integer
            The unique ID associated with a specific team
        """
        return self.model.docvecs[str(tid)]

    def load_model(self, modelfile, includeTeams=False):
        """Loads the TeamFormation gensim model
        Parameters
        ----------
        modelfile : String
            The t2v embedding model path
        includeTeams : Boolean
            Whether to include the teams or not
        """
        self.model = gensim.models.Doc2Vec.load(modelfile)  # load the doc2vec model
        if includeTeams:
            with open(modelfile.replace('model', 'teams'), 'rb') as f:
                self.teams = pickle.load(f)

    def generate_embeddings(self):
        """Generate embeddings for the provided database
        Utilize init and train methods to generate user/
        skill embeddings
        """
        min_skill_size = 0
        min_member_size = 0

        if dblp.preprocessed_dataset_exist(self.databasePath):
            return True
            team_matrix = dblp.load_preprocessed_dataset(self.databasePath)  # load the preprocessed dataset

            help_str = 'team2vec.py [-m] [-s] [-d <dimension=100>] [-e <epochs=100>] [-w <window=2>] \n-m: distributed memory mode; default=distributed bag of members\n-s: member type = skill; default = user'
            try:
                opts, args = getopt.getopt(sys.argv[1:], "hmsd:w:", ["dimension=", "window="])
            except getopt.GetoptError:
                print(help_str)
                sys.exit(2)
            dimension = 100
            epochs = 100
            window = 2
            dm = 0
            member_type = 'skill'  # to generate skill embeddings
            # member_type = 'user'
            for opt, arg in opts:
                if opt == '-h':
                    print(help_str)
                    sys.exit()
                elif opt == '-s':
                    member_type = 'skill'
                elif opt == '-m':
                    dm = 1
                elif opt in ("-d", "--dimension"):
                    dimension = int(arg)
                elif opt in ("-e", "--epochs"):
                    epochs = int(arg)
                elif opt in ("-w", "--window"):
                    window = int(arg)

            self.init(team_matrix, member_type=member_type)
            self.train(dimension=dimension, window=window, dist_mode=dm, output=self.embeddings_save_path, epochs=epochs)
            #return True

        else:
            print("The preprocessed database provided does not exist!")
            return False
