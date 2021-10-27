import pandas as pd
from time import time
from tqdm import tqdm

from cmn.team import Team
from cmn.inventor import Inventor

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
            patents = pd.read_csv(datapath, sep='\t', header=0, usecols=['id', 'type', 'country', 'date', 'title', 'withdrawn'], low_memory=False)#withdrawn may imply success or failure
            patents.rename(columns={'id': 'patent_id', 'country':'patent_country'}, inplace=True)
            patents = patents[patents['type'].isin(['utility', ''])]

            print("Reading patents' subgroups ...")
            patents_cpc = pd.read_csv(datapath.replace('patent', 'cpc_current'), sep='\t', usecols=['patent_id', 'subgroup_id', 'sequence'])
            patents_cpc.sort_values(by=['patent_id', 'sequence'], inplace=True)
            patents_cpc.reset_index(drop=True, inplace=True)
            patents_cpc = patents_cpc.groupby(['patent_id'])['subgroup_id'].apply(','.join).reset_index()
            patents_cpc = pd.merge(patents, patents_cpc, on='patent_id', how='inner', copy=False)

            #TODO: filter the patent based on subgroup e.g., cpc_subgroup: "Y10S706/XX"	"Data processing: artificial intelligence"

            print("Reading patents' inventors ...")
            patents_inventors = pd.read_csv(datapath.replace('patent', 'patent_inventor'), sep='\t', header=0)
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

