import os, scipy.sparse, pickle, multiprocessing
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import lil_matrix
import numpy as np
from time import time
from functools import partial
import pandas as pd
from dateutil import parser

class Team(object):
    def __init__(self, id, members, skills, datetime, location=None):
        self.id = int(id)
        self.datetime = parser.parse(str(datetime)).year if not pd.isna(datetime) else None
        self.members = members
        self.skills = skills
        self.location = location
        self.members_locations = [] #[(city, state, country)] the living addresses of the members. might not be the same as the address of the team (venue of a paper or country that a patent is issued)
        self.members_details = []

    def get_one_hot(self, s2i, c2i, l2i, location_type):
        # Generating one hot encoded vector for skills of team
        skill_vec_dim = len(s2i)
        x = np.zeros((1, skill_vec_dim))
        for field in self.skills: x[0, s2i[field]] = 1

        # Generating one hot encoded vector for members of team
        candidate_vec_dim = len(c2i)
        y = np.zeros((1, candidate_vec_dim))
        idnames = [f'{m.id}_{m.name}' for m in self.members]
        for idname in idnames:
            y[0, c2i[idname]] = 1
        id = np.zeros((1, 1))
        id[0, 0] = self.id

        # Generating one hot encoded vector for locations of team members
        loc_vec_dim = len(l2i)
        z = np.zeros((1, loc_vec_dim))
        for loc in self.members_locations:
            loc = loc[0] + loc[1] + loc[2] if location_type == 'city' else \
                  loc[1] + loc[2] if location_type == 'state' else \
                  loc[2]  # if location_type == 'country'
            if loc in l2i.keys():
                z[0, l2i[loc]] = 1

        return np.hstack([id, x, z, y])

    @staticmethod
    def build_index_location(teams, location_type):
        print('Indexing locations ...')
        idx = 0; l2i = {}; i2l = {};
        for t in teams:
            for loc in t.members_locations:
                loc = loc[0] + loc[1] + loc[2] if location_type == 'city' else \
                      loc[1] + loc[2] if location_type == 'state' else \
                      loc[2] #if location_type == 'country'
                if loc not in l2i.keys():
                    l2i[loc] = idx
                    i2l[idx] = loc
                    idx += 1

        return i2l, l2i

    @staticmethod
    def build_index_candidates(teams):
        print('Indexing members ...')
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
        print('Indexing skills ...')
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
        print('Indexing teams ...')
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

        for k in list(teams.keys()):
            if pd.isna(teams[k].datetime): del teams[k]

        teams = sorted(teams.values(), key=lambda x: x.datetime)

        year_idx = [] #e.g, [(0, 1900), (6, 1903), (14, 1906)] => the i shows the starting index for movies of the year
        start_year = None
        for i, v in enumerate(teams):
            if v.datetime != start_year:
                year_idx.append((i, v.datetime))
                start_year = v.datetime

        # build indexes
        indexes = {}
        indexes['i2c'], indexes['c2i'] = Team.build_index_candidates(teams)
        indexes['i2s'], indexes['s2i'] = Team.build_index_skills(teams)
        indexes['i2t'], indexes['t2i'] = Team.build_index_teams(teams)
        indexes['i2l'], indexes['l2i'] = Team.build_index_location(teams, settings["location_type"])
        indexes['i2y'] = year_idx
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
    def bucketing(bucket_size, s2i, c2i, l2i, location_type, teams):
        skill_vec_dim = len(s2i)
        candidate_vec_dim = len(c2i)
        location_vec_dim = len(l2i)
        data = lil_matrix((len(teams), 1 + skill_vec_dim + candidate_vec_dim + location_vec_dim), dtype='u1')
        data_ = np.zeros((bucket_size, 1 + skill_vec_dim + candidate_vec_dim + location_vec_dim), dtype='u1')
        j = -1
        st = time()
        for i, team in enumerate(teams):
            try:
                j += 1
                data_[j] = team.get_one_hot(s2i, c2i, l2i, location_type)
            except IndexError as ex:
                s = int(((i / bucket_size) - 1) * bucket_size)
                e = int(s + bucket_size)
                data[s: e] = data_
                j = 0
                data_[j] = team.get_one_hot(s2i, c2i, l2i, location_type)
            except Exception as ex:
                raise ex

            # if (i % bucket_size == 0): print(f'Loading {i}/{len(teams)} instances by {multiprocessing.current_process()}! {time() - st}')

        if j > -1: data[-(j+1):] = data_[0:j+1]
        return data

    @classmethod
    def generate_sparse_vectors(cls, datapath, output, filter, settings):
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
            if settings['parallel']:
                with multiprocessing.Pool() as p:
                    n_core = multiprocessing.cpu_count() if settings['ncore'] <= 0 else settings['ncore']
                    subteams = np.array_split(teams, n_core)
                    func = partial(Team.bucketing, settings['bucket_size'], indexes['s2i'], indexes['c2i'], indexes['l2i'], settings['location_type'])
                    data = p.map(func, subteams)
                    #It took 12156.825613975525 seconds to generate and store the sparse matrices of size (1729691, 818915) at ./../data/preprocessed/uspt/patent.tsv.filtered.mt5.ts3/teamsvecs.pkl
                    #It took 11935.809179782867 seconds to generate and store the sparse matrices of size (661335, 1444501) at ./../data/preprocessed/gith/data.csv/teamsvecs.pkl
            # serial
            else:
                data = Team.bucketing(settings['bucket_size'], indexes['s2i'], indexes['c2i'], teams)
            data = scipy.sparse.vstack(data, 'lil')#{'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, By default an appropriate sparse matrix format is returned!!
            vecs = {'id': data[:, 0], 'skill': data[:, 1:len(indexes['s2i']) + 1], 'loc': data[:, len(indexes['s2i']) + 1: len(indexes['s2i']) + 1 + len(indexes['l2i'])], 'member': data[:, - len(indexes['c2i']):]}

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
    def get_stats(cls, teamsvecs, obj, output, cache=True, plot=False, plot_title=None):
        try:
            print("Loading the stats pickle ...")
            if not cache: raise FileNotFoundError
            with open(f'{output}/stats.pkl', 'rb') as infile:
                stats = pickle.load(infile)
                if plot: Team.plot_stats(stats, output, plot_title)
                return stats

        except FileNotFoundError:
            print("File not found! Generating stats ...")
            stats = {}
            teamids, skillvecs, membervecs = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member']

            stats['*nteams'] = teamids.shape[0]
            stats['*nmembers'] = membervecs.shape[1] #unique members
            stats['*nskills'] = skillvecs.shape[1]

            #declaring location
            locationvecs = teamsvecs['location']
            #number of locations
            stats['*nlocation'] = locationvecs.shape[1]
            
            # how many teams have only 1 location, 2 locations, ...
            row_sums = locationvecs.sum(axis=1)
            col_sums = locationvecs.sum(axis=0)
            nteams_nlocation  = Counter(col_sums.A1.astype(int))
            stats['*nteams_nlocation1'] = {k: v for k, v in sorted(nteams_nlocation.items(), key=lambda item: item[1], reverse=True)}
            
            stats['*nmembers_nlocation1'] = {k: v for k, v in sorted(nteams_nlocation.items(), key=lambda item: item[1], reverse=True)}
            stats['*nskills_nlocation1'] = {k: v for k, v in sorted(nteams_nlocation.items(), key=lambda item: item[1], reverse=True)}


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
            #year to count
            i2y,pointer,stats['nteams_year'],stats['nmembers_year'],stats['nskills_year'],stats['avg_nskills_nteams_year'],stats['avg_single_member_year'],stats['avg_nskills_nmembers_year'],stats['avg_nteams_nmembers_year'] = obj['i2y'],0,{},{},{},{},{},{},{}
            for x in range(len(i2y)-1): stats['nteams_year'][i2y[x][1]] = i2y[1+x][0]-i2y[x][0]
            stats['nteams_year'][i2y[-1][1]] = stats['*nteams']-i2y[-1][0]
            for x in range(len(i2y)-1):
                members,skills = 0,0
                while pointer < i2y[x+1][0]:
                    if pointer in obj['i2c'] : members += 1
                    if pointer in obj['i2s']: skills += 1
                    pointer+=1
                stats['nmembers_year'][i2y[x][1]] = members
                stats['nskills_year'][i2y[x][1]] = skills
                stats['avg_nskills_nteams_year'][i2y[x][1]] = skills/stats['nteams_year'][i2y[x][1]]
                stats['avg_single_member_year'][i2y[x][1]] = list(stats['nmembers_year'].values()).count(1)/stats['nteams_year'][i2y[x][1]]
                stats['avg_nskills_nmembers_year'][i2y[x][1]] = skills/members if members != 0 else 0#list(stats['nmembers_year'].values()).count(1) if list(stats['nmembers_year'].values()).count(1) != 0 else 0
                stats['avg_nteams_nmembers_year'][i2y[x][1]] = stats['nteams_year'][i2y[x][1]]/members if members != 0 else 0#list(stats['nmembers_year'].values()).count(1) if list(stats['nmembers_year'].values()).count(1) != 0 else 0

            print(stats)
            #TODO: skills_years (2-D image)
            #TODO: candidate_years (2-D image)
            
            #how many members are from a particular location, .... 
            row_sums_ml = locationvecs.sum(axis=1)
            col_sums_ml = locationvecs.sum(axis=0)
            nmembers_nlocation  = Counter(row_sums_ml.A1.astype(int))
            stats['*nmembers_nlocation'] = {k: v for k, v in sorted(nmembers_nlocation.items(), reverse=True)}
            
            
            #how many skills are from a particular location, ....
            row_sums_sl = locationvecs.sum(axis=1)
            col_sums_sl = locationvecs.sum(axis=0)
            nskills_nlocation  = Counter(row_sums_sl.A1.astype(int))
            stats['*nskills_nlocation'] = {k: v for k, v in sorted(nskills_nlocation.items(), reverse=True)}
            
            
            #average number of people from each location
            stats['*avg_nmembers_location'] = row_sums_ml.mean()
            stats['*avg_nskills_location'] = row_sums_sl.mean()


            with open(f'{output}/stats.pkl', 'wb') as outfile: pickle.dump(stats, outfile)
            if plot: Team.plot_stats(stats, output, plot_title)
        return stats

    @staticmethod
    def plot_stats(stats, output, plot_title):
        plt.rcParams.update({'font.family': 'Consolas'})
        for k, v in stats.items():
            if '*' in k:
                print(f'{k} : {v}')
                continue
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(1, 1, 1)
            ax.loglog(*zip(*stats[k].items()), marker='x', linestyle='None', markeredgecolor='b')
            ax.set_xlabel(k.split('_')[1][0].replace('n', '#') + k.split('_')[1][1:])
            ax.set_ylabel(k.split('_')[0][0].replace('n', '#') + k.split('_')[0][1:])
            ax.grid(True, color="#93a1a1", alpha=0.3)
            # ax.spines['right'].set_color((.8, .8, .8))
            # ax.spines['top'].set_color((.8, .8, .8))
            ax.minorticks_off()
            ax.xaxis.set_tick_params(size=2, direction='in')
            ax.yaxis.set_tick_params(size=2, direction='in')
            ax.xaxis.get_label().set_size(12)
            ax.yaxis.get_label().set_size(12)
            ax.set_title(plot_title, x=0.7, y=0.8, fontsize=11)
            ax.set_facecolor('whitesmoke')
            fig.savefig(f'{output}/{k}.pdf', dpi=100, bbox_inches='tight')
            plt.show()

    @staticmethod
    def get_unigram(membervecs): return membervecs.sum(axis=0)/membervecs.shape[0]

    # needs code review
    # def loc_heatmap_skills(dataset, output):
    #     from mpl_toolkits.mplot3d import Axes3D
    #     dname = dataset.split('/')[-2]
    #
    #     # getting the data
    #     with open(f'{output}/stats.pkl', 'rb') as infile: stats = pickle.load(infile)
    #     location = stats['location']
    #     skills = stats['skills']
    #     npatents = len(stats['patent'])
    #
    #     # creating 3d figures
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = Axes3D(fig)
    #     axM = Axes3D(fig)
    #
    #     # configuring colorbar
    #     color_map = cm.ScalarMappable(cmap=cm.gray)
    #     color_map.set_array(colo)
    #
    #     # creating the heatmap
    #     img = ax.scatter(location, skills, npatents, marker='s', s=100, color='gray')
    #     plt.colorbar(color_map)
    #
    #     # adding title and labels - skills
    #     ax.set_title("3D Heatmap")
    #     ax.set_xlabel('X-locations')
    #     ax.set_ylabel('Y-skills')
    #     ax.set_zlabel('Z-number-of-patents')
    #
    #     # displaying plot
    #     fig.savefig(f"{dataset}/{dname}_location_skills_distribution.pdf", dpi=100, bbox_inches='tight')
    #     plt.show()
    #
    # def loc_heatmap_members(dataset, output):
    #     from mpl_toolkits.mplot3d import Axes3D
    #     dname = dataset.split('/')[-2]
    #     # getting the data
    #     with open(f'{output}/stats.pkl', 'rb') as infile: stats = pickle.load(infile)
    #     location = stats['location']
    #     members = stats['member']
    #     npatents = len(stats['patent'])
    #
    #     # creating 3d figures
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = Axes3D(fig)
    #     axM = Axes3D(fig)
    #
    #     # configuring colorbar
    #     color_map = cm.ScalarMappable(cmap=cm.gray)
    #     color_map.set_array(colo)
    #
    #     # creating the heatmap
    #     img = ax.scatter(location, members, npatents, marker='s', s=100, color='gray')
    #     plt.colorbar(color_map)
    #
    #     # adding title and labels - members
    #     ax.set_title("3D Heatmap")
    #     ax.set_xlabel('X-locations')
    #     ax.set_ylabel('Y-members')
    #     ax.set_zlabel('Z-number-of-patents')
    #
    #     # displaying plot
    #     fig.savefig(f"{dataset}/{dname}_location_members_distribution.pdf", dpi=100, bbox_inches='tight')
    #     plt.show()
