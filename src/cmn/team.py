import os
from scipy.sparse import lil_matrix
import scipy.sparse
import numpy as np
from time import time
import pickle
import multiprocessing
from functools import partial

class Team(object):
    def __init__(self, id, members, skills):
        self.id = id
        self.members = members
        self.skills = skills

    def get_one_hot(self, s2i, c2i):
        # Generating one hot encoded vector for skills of team
        skill_vec_dim = len(s2i)
        X = np.zeros((1, skill_vec_dim))
        for field in self.skills: X[0, s2i[field]] = 1

        # Generating one hot encoded vector for members of team
        candidate_vec_dim = len(c2i)
        y = np.zeros((1, candidate_vec_dim))
        idnames = [f'{m.id}_{m.name}' for m in self.members]
        for idname in idnames:
            y[0, c2i[idname]] = 1
        id = np.zeros((1,1))
        id[0,0] = self.id
        return np.hstack([id, X, y])

    @staticmethod
    def build_index_candidates(teams):
        idx = 0; c2i = {}; i2c = {}
        for team in teams:
            for candidate in team.members:
                idname = f'{candidate.id}_{candidate.name}'
                if idname not in c2i:
                    i2c[idx] = idname
                    c2i[idname] = idx
                    idx += 1
        return i2c, c2i

    @staticmethod
    def build_index_skills(teams):
        idx = 0; s2i = {}; i2s = {}
        for team in teams:
            for skill in team.skills:
                if skill not in s2i:
                    s2i[skill] = idx
                    i2s[idx] = skill
                    idx += 1
        return i2s, s2i

    @staticmethod
    def build_index_teams(teams):
        t2i = {}; i2t = {}
        for idx, t in enumerate(teams):
            i2t[idx] = t.id
            t2i[t.id] = idx
        return i2t, t2i

    @staticmethod
    def read_data(data_path, output, topn=None):
        # should be overridden by the children classes, customize their loading data
        # read data from file
        # build indexes
        # i2c, c2i = Team.build_index_candidates(candidates)
        # i2s, s2i = Team.build_index_skills(teams)
        # persist them as pickles
        # return i2c, c2i, i2s, s2i, teams
        pass

    @staticmethod
    def bucketing(bucket_size, s2i, c2i, teams):
        skill_vec_dim = len(s2i)
        candidate_vec_dim = len(c2i)
        data = lil_matrix((len(teams), 1 + skill_vec_dim + candidate_vec_dim))
        data_ = np.zeros((bucket_size, 1 + skill_vec_dim + candidate_vec_dim))
        j = -1
        st = time()
        for i, team in enumerate(teams):
            try:
                j += 1
                data_[j] = team.get_one_hot(s2i, c2i)
            except IndexError as ex:
                s = int(((i / bucket_size) - 1) * bucket_size)
                e = int(s + bucket_size)
                data[s: e] = data_
                j = 0
                data_[j] = team.get_one_hot(s2i, c2i)
            except Exception as ex:
                raise ex

            if (i % bucket_size == 0): print(f'Loading {i}/{len(teams)} instances! {time() - st}')

        if j > -1: data[-j:] = data_[0:j]
        return data

    @classmethod
    def generate_sparse_vectors(cls, datapath, output, topn=None, ncores=-1):
        try:
            st = time()
            print("Loading the sparse matrices...")
            with open(f'{output}/teamsvecs.pkl', 'rb') as infile: teamids, skill_vecs, member_vecs = pickle.load(infile)
            print(f"It took {time() - st} seconds to load the sparse matrices.")
            i2c, c2i, i2s, s2i, i2t, t2i, _ = cls.read_data(datapath, output, True, topn)
            return teamids, skill_vecs, member_vecs, i2c, c2i, i2s, s2i, i2t, t2i
        except FileNotFoundError as e:
            print("File not found! Generating the sparse matrices...")
            i2c, c2i, i2s, s2i, i2t, t2i, teams = cls.read_data(datapath, output, False, topn)
            st = time()
            with multiprocessing.Pool() as p:
                n_core = multiprocessing.cpu_count() if ncores < 0 else ncores
                subteams = np.array_split(list(teams.values()), n_core)
                func = partial(Team.bucketing, 1000, s2i, c2i)
                data = p.map(func, subteams)
            # data = Team.bucketing(1000, s2i, c2i, teams.values())
            data = scipy.sparse.vstack(data, 'csr')
            teamids = data[:, 0]
            skill_vecs = data[:, 1:len(s2i)]
            member_vecs = data[:, - len(c2i):]
            with open(f'{output}/teamsvecs.pkl', 'wb') as outfile: pickle.dump((teamids, skill_vecs, member_vecs), outfile)
            print(f"It took {time() - st} seconds to generate and store the sparse matrices of size {data.shape}")
            return teamids, skill_vecs, member_vecs, i2c, c2i, i2s, s2i, i2t, t2i

        except Exception as e:
            raise e

    @classmethod
    def get_stats(cls, teams, output):
        pass

