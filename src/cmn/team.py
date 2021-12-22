import os
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import lil_matrix
import scipy.sparse
import numpy as np
from time import time
import pickle
import multiprocessing
from functools import partial

class Team(object):
    def __init__(self, id, members, skills, datetime):
        self.id = id
        self.datetime = datetime
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
    def read_data(teams, output, filter, settings):
        # should be overridden by the children classes, customize their loading data
        # read data from file
        # apply filtering
        if filter: teams = Team.remove_outliers(teams, settings)
        # build indexes
        indexes = {}
        indexes['i2c'], indexes['c2i'] = Team.build_index_candidates(teams.values())
        indexes['i2s'], indexes['s2i'] = Team.build_index_skills(teams.values())
        indexes['i2t'], indexes['t2i'] = Team.build_index_teams(teams.values())
        st = time()

        try: os.makedirs(output)
        except FileExistsError as ex: pass

        with open(f'{output}/teams.pkl', "wb") as outfile: pickle.dump(teams, outfile)
        with open(f'{output}/indexes.pkl', "wb") as outfile: pickle.dump(indexes, outfile)
        print(f"It took {time() - st} seconds to pickle the data into {output}")
        return indexes, teams

    @staticmethod
    def load_data(output, index):
        st = time()
        print(f"Loading indexes pickle from {output}/indexes.pkl ...")
        with open(f'{output}/indexes.pkl', 'rb') as infile: indexes = pickle.load(infile)
        print(f"It took {time() - st} seconds to load from the pickles.")
        teams = None
        if not index:
            st = time()
            print(f"Loading teams pickle from {output}/teams.pkl ...")
            with open(f'{output}/teams.pkl', 'rb') as tfile: teams = pickle.load(tfile)
            print(f"It took {time() - st} seconds to load from the pickles.")

        return indexes, teams

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

            if (i % bucket_size == 0): print(f'Loading {i}/{len(teams)} instances by {multiprocessing.current_process()}! {time() - st}')

        if j > -1: data[-(j+1):] = data_[0:j+1]
        return data

    @classmethod
    def generate_sparse_vectors(cls, datapath, output, filter, settings):
        output += f'.filtered.mt{settings["filter"]["min_nteam"]}.ts{settings["filter"]["min_team_size"]}' if filter else ""
        pkl = f'{output}/teamsvecs.pkl'
        try:
            st = time()
            print(f"Loading sparse matrices from {pkl} ...")
            with open(pkl, 'rb') as infile: vecs = pickle.load(infile)
            indexes, _ = cls.read_data(datapath, output, index=True, filter=filter, settings=settings)
            print(f"It took {time() - st} seconds to load the sparse matrices.")
            return vecs, indexes
        except FileNotFoundError as e:
            print("File not found! Generating the sparse matrices ...")
            indexes, teams = cls.read_data(datapath, output, index=False, filter=filter, settings=settings)
            st = time()
            # parallel
            with multiprocessing.Pool() as p:
                n_core = multiprocessing.cpu_count() if settings['ncore'] <= 0 else settings['ncore']
                subteams = np.array_split(list(teams.values()), n_core)
                func = partial(Team.bucketing, settings['bucket_size'], indexes['s2i'], indexes['c2i'])
                data = p.map(func, subteams)
            # serial
            # data = Team.bucketing(1000, s2i, c2i, teams.values())
            data = scipy.sparse.vstack(data, 'lil')#{'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, By default an appropriate sparse matrix format is returned!!
            vecs = {'id': data[:, 0], 'skill': data[:, 1:len(indexes['s2i']) + 1], 'member':data[:, - len(indexes['c2i']):]}

            with open(pkl, 'wb') as outfile: pickle.dump(vecs, outfile)
            print(f"It took {time() - st} seconds to generate and store the sparse matrices of size {data.shape} at {pkl}")
            return vecs, indexes

        except Exception as e:
            raise e

    @staticmethod
    def remove_outliers(teams, settings):
        print(f'Removing outliers {settings["filter"]} ...')
        for id in list(teams.keys()):
            teams[id].members = [member for member in teams[id].members if len(member.teams) > settings['filter']['min_nteam']]
            if len(teams[id].members) < settings['filter']['min_team_size']: del teams[id]

        return teams

    @classmethod
    def get_stats(cls, teamsvecs, output, plot=False):
        try:
            print("Loading the stats pickle ...")
            with open(f'{output}/stats.pkl', 'rb') as infile:
                stats = pickle.load(infile)
                if plot: Team.plot_stats(stats, output)
                return stats

        except FileNotFoundError:
            print("File not found! Generating stats ...")
            stats = {}
            teamids, skillvecs, membervecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member']

            stats['*nteams'] = teamids.shape[0]
            stats['*nmembers'] = membervecs.shape[1] #unique members
            stats['*nskills'] = skillvecs.shape[1]

            #distributions
            row_sums = skillvecs.sum(axis=1)
            col_sums = skillvecs.sum(axis=0)
            nteams_nskills = Counter(row_sums.A1.astype(int))
            stats['nteams_nskills'] = {k: v for k, v in sorted(nteams_nskills.items(), key=lambda item: item[1], reverse=True)}
            stats['nteams_skill-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
            stats['*avg_nskills_team'] = row_sums.mean()
            stats['*nteams_single_skill'] = stats['nteams_nskills'][1] if 1 in stats['nteams_nskills'] else 0
            # how many skills have only 1 team, 2 teams, ...
            nskills_nteams = Counter(col_sums.A1.astype(int))
            stats['nskills_nteams'] = {k: v for k, v in sorted(nskills_nteams.items(), key=lambda item: item[1], reverse=True)}
            stats['*avg_nskills_member'] = ((skillvecs.transpose() @ membervecs) > 0).sum(axis=0).mean()

            row_sums = membervecs.sum(axis=1)
            col_sums = membervecs.sum(axis=0)
            nteams_nmembers = Counter(row_sums.A1.astype(int))
            stats['nteams_nmembers'] = {k: v for k, v in sorted(nteams_nmembers.items(), key=lambda item: item[1], reverse=True)}
            stats['nteams_candidate-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
            stats['*avg_nmembers_team'] = row_sums.mean()
            stats['*nteams_single_member'] = stats['nteams_nmembers'][1] if 1 in stats['nteams_nmembers'] else 0
            #how many members have only 1 team, 2 teams, ....
            nmembers_nteams = Counter(col_sums.A1.astype(int))
            stats['nmembers_nteams'] = {k: v for k, v in sorted(nmembers_nteams.items(), key=lambda item: item[1], reverse=True)}
            stats['*avg_nteams_member'] = col_sums.mean()

            #TODO: temporal stats!
            #TODO: skills_years (2-D image)
            #TODO: candidate_years (2-D image)
            with open(f'{output}/stats.pkl', 'wb') as outfile: pickle.dump(stats, outfile)
            if plot: Team.plot_stats(stats, output)
        return stats

    @staticmethod
    def plot_stats(stats, output):
        for k, v in stats.items():
            if '*' in k:
                print(f'{k} : {v}')
                continue
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_subplot(1, 1, 1)
            ax.loglog(*zip(*stats[k].items()), marker='x', linestyle='None')
            ax.set_xlabel(k.split('_')[1].replace('n', '#', 0))
            ax.set_ylabel(k.split('_')[0].replace('n', '#', 0))
            ax.grid(True, color="#93a1a1", alpha=0.3)
            ax.spines['right'].set_color((.8, .8, .8))
            ax.spines['top'].set_color((.8, .8, .8))
            ax.minorticks_off()
            ax.xaxis.set_tick_params(size=1)
            ax.yaxis.set_tick_params(size=1)
            ax.xaxis.get_label().set_size(12)
            ax.yaxis.get_label().set_size(12)
            fig.savefig(f'{output}/{k}.png', dpi=100, bbox_inches='tight')
            plt.show()

    @staticmethod
    def get_unigram(membervecs):
        return membervecs.sum(axis=0)/membervecs.shape[0]
