from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import numpy as np
from datetime import datetime
import time
import pickle


class Team(object):
    count = 0
    def __init__(self, id, members):
        Team.count += 1
        self.id = id
        self.members = members
    
    # Return a list of members' names    ==> [m.get_name() for m in team.members]

    # Return a list of members' ids ==> [m.get_id() for m in team.members]

    def get_skills(self):
        pass

    def get_id(self):
        return self.id

    @staticmethod
    def build_index_members(all_members):
        idx = 0
        m2i = {}
        i2m = {}
        start_time = time.time()
        for member in all_members.values():
            i2m[idx] = f'{member.get_id()}_{member.get_name()}'
            m2i[f'{member.get_id()}_{member.get_name()}'] = idx
            idx += 1
        print(f"It took {time.time() - start_time} seconds to build i2m and m2i.")
        return i2m, m2i

    @staticmethod
    def build_index_skills(teams):
        idx = 0
        s2i = {}
        i2s = {}
        start_time = time.time()
        for team in teams.values():
            for skill in team.get_skills():
                if skill not in s2i.keys():
                    s2i[skill] = idx
                    i2s[idx] = skill
                    idx += 1
        print(f"It took {time.time() - start_time} seconds to build i2s and s2i.")
        return i2s, s2i

    @staticmethod
    def read_data(data_path, output, topn=None):
        # should be overridden by the children classes, customize their loading data
        # read data from file
        # build indexes
        # i2m, m2i = Team.build_index_members(all_members)
        # i2s, s2i = Team.build_index_skills(teams)
        # persist them as pickles
        # return i2m, m2i, i2s, s2i, teams
        pass

    @classmethod
    def generate_sparse_vectors(cls, raw_data_path, output, topn=None):
        i2m, m2i, i2s, s2i, teams = cls.read_data(raw_data_path, output, topn)
        len_teams = len(teams)
        try:
            start_time = time.time()
            print("Loading the sparse matrices...")
            with open(f'{output}/sparse.pkl', 'rb') as infile:
                print("Loading sparse pickle...")
                skill_vecs, member_vecs = pickle.load(infile)
            print(f"It took {time.time() - start_time} seconds to load the sparse matrices.")
            return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, len_teams
        except:
            print("File not found! Generating the sparse matrices...")
            start_time = time.time()
            bucket_size = 100
            skill_size = len(s2i)
            author_size = len(m2i)
            # Sparse Matrix and bucketing
            data = lil_matrix((len_teams, skill_size + author_size))
            data_ = np.zeros((bucket_size, skill_size + author_size))
            j = -1
            for i, team in enumerate(teams.values()):
                if i >= len_teams: break

                # Generating one hot encoded vector for input
                X = np.zeros((1, skill_size))
                input_fields = team.get_skills()
                for field in input_fields:
                    X[0, s2i[field]] = 1

                # This does not work since the number of authors are different for each sample, therefore we need to build the output as a one hot encoding
                # y_index = []
                # for id in output_ids:
                #     y_index.append(m2i[id])
                # y_index.append(len(output_ids))
                # y = np.asarray([y_index])

                # Generating one hot encoded vector for output
                y = np.zeros((1, author_size))
                output_ids = [f'{m.get_id()}_{m.get_name()}' for m in team.members]
                for id in output_ids:
                    y[0, m2i[id]] = 1

                # Building a training instance
                X_y = np.hstack([X, y])

                # Bucketing
                try:
                    j += 1
                    data_[j] = X_y
                except:
                    s = int(((i / bucket_size) - 1) * bucket_size)
                    e = int(s + bucket_size)
                    data[s: e] = data_
                    j = 0
                    data_[j] = X_y
                if (i % bucket_size == 0):
                    print(f'Loading {i}/{len(teams)} instances! {datetime.now()}')
                    print(f'{time.time() - start_time} seconds has passed until now.')
            if j > -1:
                data[-j:] = data_[0:j]

            skill_vecs = data[:, :skill_size]
            member_vecs = data[:, - author_size:]
            with open(f'{output}/sparse.pkl', 'wb') as outfile:
                pickle.dump((skill_vecs, member_vecs), outfile)
            print(f"It took {time.time() - start_time} seconds to generate and store the sparse matrices.")

        return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, len_teams

    @classmethod
    def get_stats(cls, teams, output):
        pass

