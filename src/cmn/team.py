from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import numpy as np
from datetime import datetime
import time
import pickle
import multiprocessing
from itertools import zip_longest
from functools import partial

num_proc = 4


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def something(teams, bucket_size, skill_size, author_size, s2i, m2i):
    # Sparse Matrix and bucketing
    data_ = np.zeros((bucket_size, skill_size + author_size))
    j = -1

    for _, team in enumerate(teams):
        try:
            # Generating one hot encoded vector for input
            X = np.zeros((1, skill_size))
            input_fields = team.get_skills()
            for field in input_fields:
                X[0, s2i[field]] = 1

            # Generating one hot encoded vector for output
            y = np.zeros((1, author_size))
            output_ids = [f'{m.get_id()}_{m.get_name()}' for m in team.members]
            for id in output_ids:
                y[0, m2i[id]] = 1

            # Building a training instance
            j += 1
            data_[j] = np.hstack([X, y])
        except:
            # queue.put(data_[:j])
            return data_[:j]
    # queue.put(data_)
    return data_

def bucketing(teams, bucket_size, len_teams, skill_size, author_size, s2i, m2i):

    number_of_slices = len(teams)//(bucket_size*num_proc) + 1
    print(number_of_slices)
    data = lil_matrix((len_teams, skill_size + author_size))
    start_time = time.time()
    for i, slice in enumerate(grouper(list(teams.values()), bucket_size*num_proc)):
        pool = multiprocessing.Pool(num_proc)
        prod_x = partial(something, bucket_size=bucket_size, skill_size=skill_size, author_size=author_size, s2i=s2i, m2i=m2i)
        data_ = pool.map(prod_x, grouper(list(slice), bucket_size))

        pool.close()
        pool.join()
        print(f"It took {time.time() - start_time} seconds to join slice #{i}!")
        # processes = []
        # queue = multiprocessing.Queue()
        # for _, bucket in enumerate(grouper(list(slice), bucket_size)):
        #     process = multiprocessing.Process(target=something, args=(bucket, queue, bucket_size, skill_size, author_size, s2i, m2i))
        #     processes.append(process)
        #     process.start()
        # for process in processes:
        #     process.join()


        for j in range(len(data_)):
        # j = 0
        # while not queue.empty():
            s = i * bucket_size * num_proc + j * bucket_size
            e = i * bucket_size * num_proc + ((j+1)*bucket_size)
            # data[s:e] = queue.get()
        #     j += 1
            data[s:e] = data_[j]

        print(f"It took {time.time() - start_time} seconds to load slice #{i}!")
    print(f"It took {time.time() - start_time} seconds to make the sparse matrix!")

    return data


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

    def get_n_members(self):
        return len(self.members)

    def set_members(self, new_members):
        self.members = new_members

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
    def build_index_teams(teams):
        idx = 0
        t2i = {}
        i2t = {}
        start_time = time.time()
        for tid in teams.keys():
            i2t[idx] = tid
            t2i[tid] = idx
            idx += 1
        print(f"It took {time.time() - start_time} seconds to build i2t and t2i.")
        return i2t, t2i

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


    # @classmethod
    # def generate_sparse_vectors(cls, data_path, output, topn=None):
    #     i2m, m2i, i2s, s2i, i2t, t2i, teams = cls.read_data(data_path, output, topn)
    #     len_teams = len(teams)
    #     try:
    #         start_time = time.time()
    #         print("Loading the sparse matrices...")
    #         with open(f'{output}/sparse.pkl', 'rb') as infile:
    #             print("Loading sparse pickle...")
    #             skill_vecs, member_vecs = pickle.load(infile)
    #         print(f"It took {time.time() - start_time} seconds to load the sparse matrices.")
    #         return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i, len_teams
    #     except:
    #         print("File not found! Generating the sparse matrices...")
    #         start_time = time.time()
    #         bucket_size = 100
    #         skill_size = len(s2i)
    #         author_size = len(m2i)
    #         # Sparse Matrix and bucketing
    #         data = lil_matrix((len_teams, skill_size + author_size))
    #         data_ = np.zeros((bucket_size, skill_size + author_size))
    #         j = -1
    #         for i, team in enumerate(teams.values()):
    #             if i >= len_teams: break
    #
    #             # Generating one hot encoded vector for input
    #             X = np.zeros((1, skill_size))
    #             input_fields = team.get_skills()
    #             for field in input_fields:
    #                 X[0, s2i[field]] = 1
    #
    #             # This does not work since the number of authors are different for each sample, therefore we need to build the output as a one hot encoding
    #             # y_index = []
    #             # for id in output_ids:
    #             #     y_index.append(m2i[id])
    #             # y_index.append(len(output_ids))
    #             # y = np.asarray([y_index])
    #
    #             # Generating one hot encoded vector for output
    #             y = np.zeros((1, author_size))
    #             output_ids = [f'{m.get_id()}_{m.get_name()}' for m in team.members]
    #             for id in output_ids:
    #                 y[0, m2i[id]] = 1
    #
    #             # Building a training instance
    #             X_y = np.hstack([X, y])
    #             print(X_y)
    #             # Bucketing
    #             try:
    #                 j += 1
    #                 data_[j] = X_y
    #             except:
    #                 s = int(((i / bucket_size) - 1) * bucket_size)
    #                 e = int(s + bucket_size)
    #                 data[s: e] = data_
    #                 j = 0
    #                 data_[j] = X_y
    #             if (i % bucket_size == 0):
    #                 print(f'Loading {i}/{len(teams)} instances! {datetime.now()}')
    #                 print(f'{time.time() - start_time} seconds has passed until now.')
    #         if j > -1:
    #             data[-j:] = data_[0:j]
    #         print(data[0].toarray())
    #         skill_vecs = data[:, :skill_size]
    #         member_vecs = data[:, - author_size:]
    #         with open(f'{output}/sparse.pkl', 'wb') as outfile:
    #             pickle.dump((skill_vecs, member_vecs), outfile)
    #         print(f"It took {time.time() - start_time} seconds to generate and store the sparse matrices.")
    #
    #     return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i, len_teams

    @classmethod
    def generate_sparse_vectors(cls, data_path, output, topn=None):
        i2m, m2i, i2s, s2i, i2t, t2i, teams = cls.read_data(data_path, output, topn)
        len_teams = len(teams)
        try:
            start_time = time.time()
            print("Loading the sparse matrices...")
            with open(f'{output}/sparse.pkl', 'rb') as infile:
                print("Loading sparse pickle...")
                skill_vecs, member_vecs = pickle.load(infile)
            print(f"It took {time.time() - start_time} seconds to load the sparse matrices.")
            return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i, len_teams
        except:
            print("File not found! Generating the sparse matrices...")
            start_time = time.time()
            bucket_size = 20
            skill_size = len(s2i)
            author_size = len(m2i)
            # Sparse Matrix and bucketing
            data = bucketing(teams, bucket_size, len_teams, skill_size, author_size, s2i, m2i)

            print(data)
            skill_vecs = data[:, :skill_size]
            member_vecs = data[:, - author_size:]
            with open(f'{output}/sparse.pkl', 'wb') as outfile:
                pickle.dump((skill_vecs, member_vecs), outfile)
            print(f"It took {time.time() - start_time} seconds to generate and store the sparse matrices.")

        return skill_vecs, member_vecs, i2m, m2i, i2s, s2i, i2t, t2i, len_teams

    @classmethod
    def get_stats(cls, teams, output):
        pass

