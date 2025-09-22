import os, scipy.sparse, pickle, numpy as np, logging
from collections import Counter
from functools import partial, reduce
log = logging.getLogger(__name__)

import pkgmgr as opentf
class Team(object):
    def __init__(self, id, members, skills, datetime, location=None):
        self.id = int(id)
        self.datetime = opentf.install_import('python-dateutil', 'dateutil', 'parser').parse(str(datetime)).year if datetime else None
        self.members = members
        self.skills = skills
        self.location = location
        self.members_locations = [] #[(city, state, country)] the living addresses of the members. might not be the same as the address of the team (venue of a paper or country that a patent is issued)
        self.members_details = []

    def get_one_hot(self, s2i, c2i, l2i, location_type):
        # Generating one hot encoded vector for skills of team
        skill_vec_dim = len(s2i)
        x = np.zeros((1, skill_vec_dim), dtype='u1')
        for field in self.skills: x[0, s2i[field]] = 1

        # Generating one hot encoded vector for members of team
        candidate_vec_dim = len(c2i)
        y = np.zeros((1, candidate_vec_dim), dtype='u1')
        idnames = [f'{m.id}_{m.name}' for m in self.members]
        for idname in idnames: y[0, c2i[idname]] = 1

        # Generating one hot encoded vector for locations of team members
        loc_vec_dim = len(l2i)
        z = np.zeros((1, loc_vec_dim), dtype='u1')
        for loc in self.members_locations:
            loc = loc[0] + loc[1] + loc[2] if location_type == 'city' else \
                  loc[1] + loc[2] if location_type == 'state' else \
                  loc[2]  # if location_type == 'country' or 'venue'
            if loc in l2i.keys(): z[0, l2i[loc]] = 1

        return np.hstack([x, y, z])

    @staticmethod
    def remove_outliers(teams, cfg):
        log.info(f'Removing outliers {cfg} ...')
        for id in list(teams.keys()):
            teams[id].members = [member for member in teams[id].members if len(member.teams) >= cfg.min_nteam]
            # this may lead to a team with empty 0 members, which will be removed next line
            if len(teams[id].members) < cfg.min_team_size: del teams[id]
            # this may lead to skills that has no teams but the index creating is after this step. It relies on teams' skills for the remained teams

        return teams

    @staticmethod
    def build_index_location(teams, location_type):
        print('Indexing locations ...')
        idx = 0; l2i = {}; i2l = {};
        for t in teams:
            for l in t.members_locations:
                loc = l[0] + l[1] + l[2] if location_type == 'city' and all(l) else \
                      l[1] + l[2] if location_type == 'state' and all(l[1:]) else \
                      l[2] if (location_type == 'country' or 'venue') and l[2] else None
                if not loc: continue
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
    def read_data(teams, output, cfg):
        # should be overridden by the children classes, customize their loading data
        # read data from file
        # apply filtering. this should be before creating (1) indexes, and (2) teamsvecs
        if 'filter' in cfg: teams = Team.remove_outliers(teams, cfg.filter)

        for k in list(teams.keys()):
            if not teams[k].datetime: del teams[k]

        teams = sorted(teams.values(), key=lambda x: x.datetime)

        year_idx = [] #e.g, [(0, 1900), (6, 1903), (14, 1906)] => the i shows the starting index for teams of the year
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
        indexes['i2l'], indexes['l2i'] = Team.build_index_location(teams, cfg.location)
        indexes['i2y'] = year_idx

        try: os.makedirs(output)
        except FileExistsError as ex: pass

        with open(f'{output}/teams.pkl', "wb") as outfile: pickle.dump(teams, outfile)
        with open(f'{output}/indexes.pkl', "wb") as outfile: pickle.dump(indexes, outfile)
        log.info(f'Teams and indexes are pickled into {output}')
        return indexes, teams

    @staticmethod
    def load_data(output, indexes_only):
        log.info(f'Loading indexes pickle from {output}/indexes.pkl ...')
        with open(f'{output}/indexes.pkl', 'rb') as infile: indexes = pickle.load(infile)
        teams = None
        if not indexes_only:
            log.info(f'Loading teams pickle from {output}/teams.pkl ...')
            with open(f'{output}/teams.pkl', 'rb') as tfile: teams = pickle.load(tfile)
        log.info('Indexes pickle is loaded.' if indexes_only else 'Teams and indexes pickles are loaded.')
        return indexes, teams

    @staticmethod
    def bucketing(bucket_size, s2i, c2i, l2i, location_type, teams):
        skill_vec_dim = len(s2i)
        candidate_vec_dim = len(c2i)
        location_vec_dim = len(l2i)

        data = scipy.sparse.lil_matrix((len(teams), skill_vec_dim + candidate_vec_dim + location_vec_dim), dtype='u1')
        data_ = np.zeros((bucket_size, skill_vec_dim + candidate_vec_dim + location_vec_dim), dtype='u1')
        j = -1
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

    @staticmethod
    def validate(teamsvecs):
        if (teamsvecs['skill'].shape[0] < 1): return (False, f'No teams in skill metrix!')

        if (teamsvecs['skill'].shape[1] < 1): return (False, f'No skills in skill metrix!')

        if (any(not row for row in teamsvecs['skill'].rows)): # teams with empty skills
            zero_row_indices = [i for i, row in enumerate(teamsvecs['skill'].rows) if not row]
            return (False, f'Following teams have no skills!\n{zero_row_indices}')

        teamsvecs_csc = teamsvecs['skill'].tocsc()
        if((teamsvecs_csc.getnnz(axis=0) == 0).any()):
            zero_col_indices = (teamsvecs_csc['skill'].getnnz(axis=0) == 0).nonzero()[0]
            return (False, f'Following skills are used in no teams!\n{zero_col_indices}')

        if (teamsvecs['member'].shape[0] < 1): return (False, f'No teams in member metrix!')

        if (teamsvecs['member'].shape[1] < 1): return (False, f'No member in member metrix!')

        if (any(not row for row in teamsvecs['member'].rows)):  # teams with empty members
            zero_row_indices = [i for i, row in enumerate(teamsvecs['member'].rows) if not row]
            return (False, f'Following teams have no members!\n{zero_row_indices}')

        teamsvecs_csc = teamsvecs['member'].tocsc()
        if ((teamsvecs_csc.getnnz(axis=0) == 0).any()):
            zero_col_indices = (teamsvecs_csc['member'].getnnz(axis=0) == 0).nonzero()[0]
            return (False, f'Following skills are used in no teams!\n{zero_col_indices}')

        if teamsvecs['loc'] is not None:
            #in dblp, the 'loc' is the replica of the paper venue, so it should be 1-hot for each team
            #in uspt, the 'loc' is the actual location of the inventor, so it can be multihot.
            e = [i for i in range(teamsvecs['loc'].shape[0]) if (len(teamsvecs['loc'].rows[i]) != 1) or (sum(teamsvecs['loc'].data[3]) != 1)]
            if e: log.warning(f'{opentf.textcolor["yellow"]}Following teams are not one-hot in the location of team members.{opentf.textcolor["reset"]} '
                              f'Based on the underlying dataset/domain, it may be valid like in uspt, or invalid like dblp.\n{e}')

        return (True, '')

    @classmethod
    def gen_teamsvecs(cls, datapath, output, cfg):
        def __load_teamsvecs_from_file(datapath, output, cfg, pkl):
            with open(pkl, 'rb') as infile: vecs = pickle.load(infile)
            indexes, _ = cls.read_data(datapath, output, cfg, indexes_only=True)
            log.info(f"Teamsvecs matrices and indexes for skills {vecs['skill'].shape}, members {vecs['member'].shape}, and locations {vecs['loc'].shape if vecs['loc'] is not None else None} are loaded.")
            assert vecs['skill'].shape[1] == len(indexes['i2s']) or \
                   vecs['member'].shape[1] == len(indexes['i2c']) or \
                  (vecs['loc'] is not None and vecs['loc'].shape[1] == len(indexes['i2l'])) , \
                f'{opentf.textcolor["red"]}Incompatible teamsvecs and indexes!{opentf.textcolor["reset"]}'
            return vecs, indexes

        pkl = f'{output}/teamsvecs.pkl'
        try:
            log.info(f"Loading teamsvecs matrices from {pkl} ...")
            return __load_teamsvecs_from_file(datapath, output, cfg, pkl)
        except FileNotFoundError as e:
            # retry to download from hugging face. The pkl file is in the same path as output, except the '../' is removed
            if( 'hf' in cfg and cfg.hf and
                opentf.get_from_hf(repo_type='dataset', filename=pkl.replace('../', '')) and
                opentf.get_from_hf(repo_type='dataset', filename=f'{output.replace("../","")}/indexes.pkl')):
                return __load_teamsvecs_from_file(datapath, output, cfg, pkl)
            
            log.info("Teamsvecs matrices and/or indexes not found! Generating ...")
            indexes, teams = cls.read_data(datapath, output, cfg, indexes_only=False)

            # there should be no difference in content of teavsvecs when using different acceleration method
            # do unit test manually as explained here: https://github.com/fani-lab/OpeNTF/issues/286#issuecomment-2920203907
            if 'acceleration' in cfg and 'cpu' in cfg.acceleration:
                import multiprocessing
                with multiprocessing.Pool() as p:
                    n_core = multiprocessing.cpu_count() if cfg.acceleration == 'cpu' else int(cfg.acceleration.split(':')[1])
                    subteams = np.array_split(teams, n_core)
                    func = partial(Team.bucketing, cfg.bucket_size, indexes['s2i'], indexes['c2i'], indexes['l2i'], cfg.location)
                    data = p.map(func, subteams)
                    #It took 12156.825613975525 seconds to generate and store the sparse matrices of size (1729691, 818915) at ./../data/preprocessed/uspt/patent.tsv.filtered.mt5.ts3/teamsvecs.pkl
                    #It took 11935.809179782867 seconds to generate and store the sparse matrices of size (661335, 1444501) at ./../data/preprocessed/gith/data.csv/teamsvecs.pkl

                data = scipy.sparse.vstack(data, 'lil')#{'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, By default an appropriate sparse matrix format is returned!!

            elif 'acceleration' in cfg and 'cuda' in cfg.acceleration:
                torch = opentf.install_import('torch')
                try: torch.tensor([1.0], device=(device := torch.device(cfg.acceleration if ':' in cfg.acceleration else 'cuda:0')))
                except RuntimeError as e: raise RuntimeError(f'{opentf.textcolor["red"]}{cfg.acceleration}-->{device} is not available or invalid!{opentf.textcolor["reset"]}') from e
                log.info(f'Using gpu {opentf.textcolor["blue"]}{cfg.acceleration}-->{device}{opentf.textcolor["reset"]} for teams vectors generation ...')

                s2i, c2i, l2i = indexes['s2i'], indexes['c2i'], indexes['l2i']
                total_dim = len(s2i) + len(c2i) + len(l2i)
                
                gpu_tensor_batches = []
                for i in range(0, len(teams), cfg.bucket_size):
                    batch = teams[i:min(i + cfg.bucket_size, len(teams))]
                    if not batch: continue
                    
                    # Process batch and send to GPU
                    one_hot_arrays = [team.get_one_hot(s2i, c2i, l2i, cfg.location) for team in batch]
                    batch_array = np.vstack(one_hot_arrays) if one_hot_arrays else np.zeros((0, total_dim), dtype='u1')
                    gpu_tensor_batches.append(torch.from_numpy(batch_array).to(device))
                
                # Convert back to sparse matrix for validation
                if gpu_tensor_batches: data = scipy.sparse.lil_matrix((torch.vstack(gpu_tensor_batches)).cpu().numpy())
                else: data = scipy.sparse.lil_matrix((0, total_dim), dtype='u1')
                
            # serial
            else: data = Team.bucketing(cfg.bucket_size, indexes['s2i'], indexes['c2i'], indexes['l2i'], cfg.location, teams)

            vecs = {'skill': data[:, :len(indexes['s2i'])],
                    'member': data[:, len(indexes['s2i']): len(indexes['s2i']) + len(indexes['c2i'])],
                    'loc': data[:, - len(indexes['l2i']):] if len(indexes['l2i']) > 0 else None}

            # unit test on toy.dblp.v12.json
            # data[:, 0:len(indexes['s2i'])].todense()[5]
            # matrix([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=uint8)
            # teams[5].skills
            # {'deep_learning', 'object_detection'}
            # indexes['i2s'][4]
            # 'object_detection'
            # indexes['i2s'][5]
            # 'deep_learning'
            # check no rows (teams) with empty skills, or empty members
            # check no columns (skills or members) with no value (no team)
            # assert Team.validate(vecs) --> not working! as a tuple is True :D
            assert (r := Team.validate(vecs))[0], f'{opentf.textcolor["red"]}{r[1]}{opentf.textcolor["reset"]}'
            with open(pkl, 'wb') as outfile: pickle.dump(vecs, outfile)
            log.info(f"Teamsvecs matrices for skills {vecs['skill'].shape}, members {vecs['member'].shape}, and locations {vecs['loc'].shape if vecs['loc'] is not None else None} saved at {pkl}")
            return vecs, indexes

        except Exception as e: raise e

    @classmethod
    def gen_skill_coverage(cls, teamsvecs, output, skipteams=None):
        '''
        a 1-hot vector containing skills that each member has in total by transposing 'member' and then doing dot product with 'skill'
        gives us the co-occurrence matrix of member vs skills. In this way we get the number of times member x co-occurs with skill y. Then,
        each row of the es_vecs will give us all the skills a member has if the matrix is like this :
        0 4 2
        0 0 1
        2 1 0
        0 5 0
        then, the skills of the members are
        e0 -> s1 (4 times), s2 (2 times)
        e1 -> s2 (1 times)
        e2 -> s0 (2 times), s1 (1 time)
        e3 -> s2 (5 times)
        '''

        filepath = f'{output}/skillcoverage.pkl'
        try :
            log.info(f'Loading member-skill co-occurrence matrix ({teamsvecs["member"].shape[1]}, {teamsvecs["skill"].shape[1]}) from {filepath} ...')
            with open(filepath, 'rb') as f: member_skill_co = pickle.load(f)
            assert member_skill_co.shape == (teamsvecs["member"].shape[1], teamsvecs["skill"].shape[1]), f'{opentf.textcolor["red"]}Incorrect matrix size!{opentf.textcolor["reset"]}'
            return member_skill_co
        except FileNotFoundError as e:
            log.info(f'Member-skill co-occurrence matrix not found! Generating ...')

            if skipteams is not None: # to avoid test/unseen teams leakage
                import copy
                member, skill = copy.deepcopy(teamsvecs['member']), copy.deepcopy(teamsvecs['skill'])
                for i in skipteams:
                    member.rows[i] = []; member.data[i] = []
                    skill.rows[i] = []; skill.data[i] = []

                member_skill_co = scipy.sparse.csr_matrix(np.dot(member.transpose(), skill))
            else: member_skill_co = scipy.sparse.csr_matrix(np.dot(teamsvecs['member'].transpose(), teamsvecs['skill']))
            with open(filepath, 'wb') as f:
                pickle.dump(member_skill_co, f)
                log.info(f'Member-skill co-occurrence matrix {member_skill_co.shape} saved at {filepath}.')
            return member_skill_co

    @classmethod
    def get_stats(cls, teamsvecs, obj, output, cache=True, plot=False, plot_title=None):
        try:
            log.info(f'Loading the stats pickle ...')
            if not cache: raise FileNotFoundError
            with open(f'{output}/stats.pkl', 'rb') as infile:
                stats = pickle.load(infile)
                if plot: Team.plot_stats(stats, output, plot_title)
                return stats

        except FileNotFoundError:
            log.info(f'File not found! Generating stats ...')
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

    @classmethod
    def merge_teams_by_skills(cls, teamsvecs, inplace=False, distinct=False): #https://github.com/fani-lab/OpeNTF/issues/156
        # teamsvecs = {}
        # 1 110 0110
        # 2 110 1110
        # ------------
        # 3 011 0111
        # 4 011 0110
        # ------------
        # 5 111 1110
        # teamsvecs['id'] = scipy.sparse.lil_matrix([[1],[2],[3],[4],[5]])
        # teamsvecs['skill'] = scipy.sparse.lil_matrix([[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]])
        # teamsvecs['member'] = scipy.sparse.lil_matrix([[0,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,0],[1,1,1,0]])

        # new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=True)
        # 1 110 1110
        # 3 011 0111
        # 5 111 1110

        # print(new_teamsvecs['skill'].todense())# <= [[1, 1, 0], [0, 1, 1], [1, 1, 1]]
        # print(new_teamsvecs['member'].todense())# <= [[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]
        #
        # new_teamsvecs = merge_teams_by_skills(teamsvecs, inplace=False, distinct=False)
        # print(new_teamsvecs['skill'].todense())# <= [[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,1,1]]
        # print(new_teamsvecs['member'].todense())# <= [[1,1,1,0],[1,1,1,0],[0,1,1,1],[0,1,1,1],[1,1,1,0]]
        # 1 110 1110
        # 2 110 1110
        # ------------
        # 3 011 0111
        # 4 011 0111
        # ------------
        # 5 111 1110

        import copy
        log.info(f'Merging teams whose subset of skills are the same ...')

        vecs = teamsvecs if inplace else copy.deepcopy(teamsvecs)
        skills_rows_map = {} # {skill: [row indices]}

        for i in range(vecs['skill'].shape[0]):
            current_skills = tuple(vecs['skill'].rows[i])
            skills_rows_map.setdefault(current_skills, []).append(i)

        del_list = []

        for _, rows in skills_rows_map.items():
            if len(rows) < 2: continue
            del_list.extend(rows[1:])
            new_members = reduce(lambda x, y: x.maximum(y), (vecs['member'].getrow(r) for r in rows))
            for row in rows: vecs['member'][row, :] = new_members
        if distinct:
            vecs['skill'] = scipy.sparse.lil_matrix(np.delete(vecs['skill'].toarray(), del_list, axis=0))
            vecs['member'] = scipy.sparse.lil_matrix(np.delete(vecs['member'].toarray(), del_list, axis=0))

        return vecs

    @staticmethod
    def plot_stats(stats, output, plot_title):
        plt = opentf.install_import('matplotlib', 'matplotlib.pyplot')
        plt.rcParams.update({'font.family': 'Consolas'})
        for k, v in stats.items():
            if '*' in k: print(f'{k} : {v}'); continue
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
