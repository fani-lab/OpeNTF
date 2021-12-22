import pandas as pd
from time import time
from tqdm import tqdm
import pickle

from cmn.team import Team
from cmn.inventor import Inventor

import pickle

class Patent(Team):
    def __init__(self, id, members, date, title, country, subgroups, withdrawn, members_details):
        super().__init__(id, members, set(subgroups.split(',')), date)
        self.title = title
        self.country = country
        self.subgroups = subgroups
        self.withdrawn = withdrawn
        self.members_details = members_details

        for i, member in enumerate(self.members):
            member.teams.append(self.id)
            member.skills.update(set(self.skills))
            member.locations.append(self.members_details[i])

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        try:
            return super(Patent, Patent).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} ...")

            #data dictionary can be find at: https://patentsview.org/download/data-download-dictionary
            print("Reading patents ...")
            patents = pd.read_csv(datapath, sep='\t', header=0, dtype={'id':'object'}, usecols=['id', 'type', 'country', 'date', 'title', 'withdrawn'], low_memory=False)#withdrawn may imply success or failure
            patents.rename(columns={'id': 'patent_id', 'country':'patent_country'}, inplace=True)
            patents = patents[patents['type'].isin(['utility', ''])]

            print("Reading patents' subgroups ...")
            patents_cpc = pd.read_csv(datapath.replace('patent', 'cpc_current'), sep='\t', dtype={'patent_id':'object'}, usecols=['patent_id', 'subgroup_id', 'sequence'])
            patents_cpc.sort_values(by=['patent_id', 'sequence'], inplace=True)#to keep the order of subgroups
            patents_cpc.reset_index(drop=True, inplace=True)
            patents_cpc = patents_cpc.groupby(['patent_id'])['subgroup_id'].apply(','.join).reset_index()
            patents_cpc = pd.merge(patents, patents_cpc, on='patent_id', how='inner', copy=False)

            #TODO: filter the patent based on subgroup e.g., cpc_subgroup: "Y10S706/XX"	"Data processing: artificial intelligence"

            print("Reading patents' inventors ...")
            patents_inventors = pd.read_csv(datapath.replace('patent', 'patent_inventor'), sep='\t', header=0, dtype={'patent_id':'object'})
            patents_cpc_inventors = pd.merge(patents_cpc, patents_inventors, on='patent_id', how='inner', copy=False)

            print("Reading inventors ...")
            inventors = pd.read_csv(datapath.replace('patent', 'inventor'), sep='\t', header=0, dtype={'male_flag':'boolean'}, usecols=['id', 'name_first', 'name_last', 'male_flag'])
            patents_cpc_inventors = pd.merge(patents_cpc_inventors, inventors, left_on='inventor_id', right_on='id', how='inner', copy=False)
            patents.rename(columns={'id': 'inv_id'}, inplace=True)

            print("Reading location data ...")
            locations = pd.read_csv(datapath.replace('patent', 'location'), sep='\t', header=0, usecols=['id', 'city', 'state', 'country'])
            patents_cpc_inventors_location = pd.merge(patents_cpc_inventors, locations, left_on='location_id', right_on='id', how='inner', copy=False)

            patents_cpc_inventors_location.sort_values(by=['patent_id'], inplace=True)
            patents_cpc_inventors_location = patents_cpc_inventors_location.append(pd.Series(), ignore_index=True)

            print("Reading data to objects..")
            teams = {}; candidates = {}; n_row = 0
            current = None

            # 100 % |██████████████████████████████████████████████████████████████████████▉ | 210808 / 210809[00:02 < 00:00,75194.06 it / s]
            for patent_team in tqdm(patents_cpc_inventors_location.itertuples(), total=patents_cpc_inventors_location.shape[0]):
                try:
                    if pd.isnull(new := patent_team.patent_id): break
                    if current != new:
                        team = Patent(patent_team.patent_id,#for "utility" patents is integer but for "design" has "Dxxxx", ...
                                      [],
                                      patent_team.date,
                                      patent_team.title,
                                      patent_team.patent_country,
                                      patent_team.subgroup_id,
                                      bool(patent_team.withdrawn),
                                      [])
                        current = new
                        teams[team.id] = team

                    inventor_id = patent_team.inventor_id
                    inventor_name = f'{patent_team.name_first}_{patent_team.name_last}'

                    if (idname := f'{inventor_id}_{inventor_name}') not in candidates:
                        candidates[idname] = Inventor(patent_team.inventor_id, inventor_name, patent_team.male_flag)
                    team.members.append(candidates[idname])
                    team.members_details.append((patent_team.city, patent_team.state, patent_team.country))

                    candidates[idname].skills.update(team.skills)
                    candidates[idname].teams.append(team.id)
                    candidates[idname].locations.append(team.members_details[-1])

                except Exception as e:
                    raise e
            return super(Patent, Patent).read_data(teams, output, filter, settings)

    @classmethod
    def get_stats(cls, teams, teamsvecs, output, plot=False):
        try:
            print("Loading the stats pickle ...")
            with open(f'{output}/stats.pkl', 'rb') as infile:
                stats = pickle.load(infile)
                if plot: Team.plot_stats(stats, output)
                return stats

        except FileNotFoundError:
            stats = {}
            stats.update(super().get_stats(teamsvecs, output, plot))

            # dic[patent.country] +=1
            # dic[patent.country, skill] +=1

            # dict[inventor.country] +=1 over inventors' locations

            # inventors' location != patent.location

            city = {}; state = {}; country = {}
            geo_loc = [city, state, country]
            loc_pat = {}; city_mem = {}; state_mem = {}; country_mem = {}
            avg_country = {}; avg_state = {}; avg_city = {}
            unq_country = set(); unq_state = set(); unq_city = set()

            for key in teams.keys():
                t_city = set(); t_state = set(); t_country = set()
                loc = teams[key].members_details[0:]
                for item in loc:
                    city_name, state_name, country_name = item
                    t_city.add(city_name); t_state.add(state_name); t_country.add(country_name)
                    if city_name in city.keys(): city[city_name] = city[city_name] + 1;
                    else:
                        city[city_name] = 1
                        unq_city.add(city_name)
                    if state_name in state.keys(): state[state_name] = state[state_name] + 1;
                    else:
                        state[state_name] = 1
                        unq_state.add(state_name)
                    if country_name in country.keys(): country[country_name] = country[country_name] + 1;
                    else:
                        country[country_name] = 1
                        unq_country.add(country_name)
                avg_city[key] = len(t_city)
                avg_state[key] = len(t_state)
                avg_country[key] = len(t_country)

            stats['npatents_avgcity'] = avg_city
            stats['npatents_avgstate'] = avg_state
            stats['npatents_avgcountry'] = avg_country
            stats['ninventors_city'] = city
            stats['ninventors_state'] = state
            stats['ninventors_country'] = country
            stats['nunique_city'] = unq_city
            stats['nunique_state'] = set(filter(lambda x: x == x , unq_state))
            stats['nunique_country'] = unq_country

            max_records = teamsvecs['id'].shape[0]
            for i in range(0, max_records):
                id = teamsvecs['id'][i].astype(int).toarray()[0][0].tolist()
                loc = teams[f'{id}'].members_details[0:]
                for loc_i in loc:
                    city_name, state_name, country_name = loc_i

                    if city_name in city_mem.keys(): city_mem[city_name] = city_mem[city_name] + 1
                    else: city_mem[city_name] = 1
                    if state_name in state_mem.keys(): state_mem[state_name] = state_mem[state_name] + 1
                    else: state_mem[state_name] = 1
                    if country_name in country_mem.keys(): country_mem[country_name] = country_mem[country_name] + 1
                    else: country_mem[country_name] = 1

            country_skill = {}
            for idx, id_r in enumerate(list(teams.keys())):
                for rec in teams[id_r].members_details:
                    _, _, country = rec
                    if country in country_skill.keys():
                        country_skill[country] = country_skill[country] + (teamsvecs['skill'][idx] != 0).sum(1)[0, 0]
                    else:
                        country_skill[country] = (teamsvecs['skill'][idx] != 0).sum(1)[0, 0]

            stats['nskills_country-idx'] = country_skill
            stats['nmembers_city-idx'] = city_mem
            stats['nmembers_state-idx'] = state_mem
            stats['nmembers_country-idx'] = country_mem
            with open(f'{output}/stats.pkl', 'wb') as outfile: pickle.dump(stats, outfile)
            if plot: Team.plot_stats(stats, output)
            return stats