import pandas as pd
import pickle


class Stats:
    def __init__(self, inventors):
        self.save_path = '../data/processed/indexes_inventor.pkl'
        self.all_inventors = inventors

    def get_stats(self):
        with open(f'{self.save_path}', "rb") as infile:
            print('Loading Pickle File.......')
            stats = pickle.load(infile)
            for item in stats:
                print(item[3])
                break
                for val in item.values():
                    print(val)
                break

            #
            # bucket_size = 100
            # skill_size = len(s2i)
            # author_size = len(m2i)
            # # Sparse Matrix and bucketing
            # data = lil_matrix((len_teams, SKILL_SIZE + AUTHOR_SIZE))
            # data_ = np.zeros((BUCKET_SIZE, SKILL_SIZE + AUTHOR_SIZE))
            # data = lil_matrix((len_teams, skill_size + author_size))
            # data_ = np.zeros((bucket_size, skill_size + author_size))





