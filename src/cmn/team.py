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
        member_to_index = {}
        index_to_member = {}
        start_time = time.time()
        for member in all_members.values():
            index_to_member[idx] = f'{member.get_id()}_{member.get_name()}'
            member_to_index[f'{member.get_id()}_{member.get_name()}'] = idx
            idx += 1
        print(f"It took {time.time() - start_time} seconds to build i2m and m2i.")
        return index_to_member, member_to_index

    @staticmethod
    def build_index_skills(teams):
        idx = 0
        skill_to_index = {}
        index_to_skill = {}
        start_time = time.time()
        for team in teams.values():
            for skill in team.get_skills():
                if skill not in skill_to_index.keys():
                    skill_to_index[skill] = idx
                    index_to_skill[idx] = skill
                    idx += 1
        print(f"It took {time.time() - start_time} seconds to build i2s and s2i.")
        return index_to_skill, skill_to_index

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
        try:
            start_time = time.time()
            print("Loading the sparse matrices...")
            skill_vecs, member_vecs = load_npz(f'{output}/sparse_vecs.npz')
            print(f"It took {time.time() - start_time} seconds to load the sparse matrices.")
            return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, teams
        except:
            print("File not found! Generating the sparse matrices...")
            start_time = time.time()
            training_size = len(teams)
            BUCKET_SIZE = 100
            SKILL_SIZE = len(s2i)
            AUTHOR_SIZE = len(m2i)

            # Sparse Matrix and bucketing
            data = lil_matrix((training_size, SKILL_SIZE + AUTHOR_SIZE))
            data_ = np.zeros((BUCKET_SIZE, SKILL_SIZE + AUTHOR_SIZE))
            j = -1
            for i, team in enumerate(teams.values()):
                if i >= training_size: break

                # Generating one hot encoded vector for input
                X = np.zeros((1, SKILL_SIZE))
                input_fields = team.get_skills()
                for field in input_fields:
                    X[0, s2i[field]] = 1

                # This does not work since the number of authors are different for each sample, therefore we need to build the output as a one hot encoding
                # y_index = []
                # for id in output_ids:
                #     y_index.append(member_to_index[id])
                # y_index.append(len(output_ids))
                # y = np.asarray([y_index])

                # Generating one hot encoded vector for output
                y = np.zeros((1, AUTHOR_SIZE))
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
                    s = int(((i / BUCKET_SIZE) - 1) * BUCKET_SIZE)
                    e = int(s + BUCKET_SIZE)
                    data[s: e] = data_
                    j = 0
                    data_[j] = X_y
                if (i % BUCKET_SIZE == 0):
                    print(f'Loading {i}/{len(teams)} instances! {datetime.now()}')
                    print(f'{time.time() - start_time} seconds has passed until now.')
            if j > -1:
                data[-j:] = data_[0:j]

            save_npz(f'{output}sparse_vecs.npz', data.tocsr())
            print(f"It took {time.time() - start_time} seconds to generate and store the sparse matrices.")

            skill_vecs = data[:, :SKILL_SIZE]
            member_vecs = data[:, - AUTHOR_SIZE:]
            return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, teams

    @classmethod
    def get_stats(cls, teams, output):
        pass

