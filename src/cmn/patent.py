import pandas as pd
from time import time
from tqdm import tqdm

import numpy as np

import pickle

import scipy.sparse as sp
import os.path
from team import Team
from inventor import Inventor


class Patent(Team):
    def __init__(self, patent_id, members, date, title, members_details):
        super().__init__(id, members, None, date)
        self.patent_id = patent_id
        self.date = date
        self.title = title
        self.members_details = members_details

        for i, member in enumerate(self.members):
            member.teams.append(self.id)
            member.skills.update(set(self.skills))
            member.role.append(self.members_details[i])

    @staticmethod
    def subset_patents(subgroup_path, cpc_path):
        try:
            final_df = pd.read_csv('../data/processed/subset_df.csv')
            return final_df
        except FileNotFoundError:
            start_time = time.time()
            cpc_chunks = pd.read_csv(cpc_path, sep='\t', chunksize=100000, usecols=['patent_id', 'subgroup_id'])
            df_concat = pd.DataFrame()
            for chunk in cpc_chunks:
                df_concat = pd.concat([df_concat, chunk[chunk['subgroup_id'].str.contains('^Y10S706', regex=True)]])
            df_concat.reset_index(drop=True)
            subgroup_df = pd.read_csv(subgroup_path, sep='\t')
            final_df = pd.merge(df_concat[['patent_id', 'subgroup_id']], subgroup_df, left_on='subgroup_id', right_on='id')
            final_df['title'] = final_df['title'].apply(lambda x: x.split('-')[-1])
            final_df['title'] = final_df['title'].apply(lambda x: x.replace("e.g.", ""))
            final_df.rename(columns={'title': 'skills'}, inplace=True)
            final_df.drop(columns=['subgroup_id', 'id'], inplace=True)
            final_df = final_df.sort_values(by=['patent_id'])
            final_df.to_csv('../data/processed/subset_df.csv', index=False)
            return final_df

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        try:
            return super(Patent, Patent).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} ...")
            # subgroup_path = datapath.replace('patent', 'cpc_subgroup')
            # cpc_path = datapath.replace('patent', 'cpc_current')
            subset_df = Patent.subset_patents(datapath.replace('patent', 'cpc_subgroup'), datapath.replace('patent', 'cpc_current'))

            print("Reading Patent Data..")
            patent_initial = pd.read_csv(datapath, sep='\t', header=0, na_values='\\N', usecols=['patent_id', 'date', 'title'])
            patent_initial = patent_initial[patent_initial['patent_id'].isin(subset_df['patent_id'])]

            print("Reading Patent_Inventor Data..")
            inv_ids = pd.read_csv(datapath.replace('patent', 'patent_inventor'), sep='\t', header=0, na_values='\\N')
            inv_ids = inv_ids[inv_ids['patent_id'].isin(subset_df['patent_id'])]

            print("Reading Inventor Data..")
            inv_details = pd.read_csv(datapath.replace('patent', 'inventor'), sep='\t', header=0, na_values='\\N',
                                      usecols=['id', 'name_first', 'name_last', 'male_flag'])
            inv_details = inv_ids[inv_ids['id'].isin(inv_ids['inventor_id'])]
            inv_details['full_name'] = inv_details['name_first'] + '_' + inv_details['name_last']
            inv_details.drop(['name_first', 'name_last'], axis=1, inplace=True)

            print("Reading Location Data..")
            loc_details = pd.read_csv(datapath.replace('patent', 'location'), sep='\t', header=0, na_values='\\N',
                                      usecols=['id', 'city', 'state', 'country'])
            loc_details = loc_details[loc_details['id'].isin(inv_details['location_id'])]
            loc_details.rename(columns={'id': 'location_id'}, inplace=True)

            pat_inv = pd.merge(patent_initial, inv_ids, left_on='patent_id', right_on='patent_id', how='inner')
            pat_inv.drop(columns='patent_id', inplace=True)
            pat_inv_details = pd.merge(pat_inv, inv_details, left_on='inventor_id', right_on='inventor_id', how='inner')
            pat_inv_details.drop(columns='inventor_id', inplace=True)
            pat_skills = pd.merge(pat_inv_details, subset_df, left_on='patent_id', right_on='patent_id', how='inner')
            pat_skills.drop(columns='patent_id', inplace=True)
            pat_all_loc = pd.merge(pat_skills, loc_details, left_on='location_id', right_on='location_id', how='inner')
            pat_all_loc.drop(colmns='location_id', inplace=True)

            pat_all_loc.dropna(inplace=True)
            pat_all_loc.sort_values(by=['patent_id'], inplace=True)
            pat_all_loc = pat_all_loc.append(pd.Series(), ignore_index=True)

            print("Reading data to objects..")
            teams = {}; candidates = {}; n_row = 0
            current = None

            for patent_team in tqdm(pat_all_loc.itertuples(), total=pat_all_loc.shape[0]):
                try:
                    if pd.isnull(new := patent_team.patent_id): break
                    if current != new:
                        team = Patent(patent_team.patent_id,
                                      [],
                                      patent_team.date,
                                      patent_team.title,
                                      [])
                        current = new
                        teams[team.patent_id] = team

                        inventor_id = patent_team.inventor_id
                        inventor_name = patent_team.full_name

                        if (idname := f'{inventor_id}_{inventor_name}') not in candidates:
                            candidates[idname] = Inventor(patent_team.inventor_id,
                                                          patent_team.location_id,
                                                          patent_team.full_name)
                        team.members.append(candidates[idname])
                        team.members_details.append((patent_team.city, patent_team.state, patent_team.country))

                        candidates[idname].skills.update(team.skills)
                        candidates[idname].teams.append(team.patent_id)
                        candidates[idname].roles.append(team.members_details[-1])

                except Exception as e:
                    raise e
            return super(Patent, Patent).read_data(teams, output, filter, settings)

