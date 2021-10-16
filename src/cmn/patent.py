import pandas as pd
import time
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import getskills as gs
import numpy as np
from cmn.author import author
import pickle
import stats as st
import scipy.sparse as sp
from cmn.team import Team
import os.path


class Assignee:
    def __init__(self, assignee_path, inventor_path, location_path, save_path, patent_path):
        self.assignee_path = assignee_path
        self.inventor_path = inventor_path
        self.location_path = location_path
        self.save_path = save_path
        self.assignee_save_location = ""
        self.inventor_save_location = ""
        # self.patent_save_location = ""
        self.assignee_location = pd.DataFrame()
        self.inventor_location = pd.DataFrame()
        self.patent_location = pd.DataFrame()
        self.patent_path = patent_path
        self.cpc_path = '../data/raw/cpc_current.tsv'
        self.inventor_details = '../data/raw/inventor.tsv'
        self.subgroup_path = '../data/raw/cpc_subgroup.tsv'
        self.inventor_f = pd.DataFrame()

    def subset_patents(self):
        try:
            final_df = pd.read_csv('../data/processed/subset_df.csv')
            return final_df
        except FileNotFoundError:
            start_time = time.time()
            cpc_chunks = pd.read_csv(self.cpc_path, sep='\t', chunksize=100000, usecols=['patent_id', 'subgroup_id'])
            df_concat = pd.DataFrame()
            for chunk in cpc_chunks:
                df_concat = pd.concat([df_concat, chunk[chunk['subgroup_id'].str.contains('^Y10S706', regex=True)]])
            df_concat.reset_index(drop=True)
            subgroup_df = pd.read_csv(self.subgroup_path, sep='\t')
            final_df = pd.merge(df_concat[['patent_id', 'subgroup_id']], subgroup_df, left_on='subgroup_id', right_on='id')
            final_df['title'] = final_df['title'].apply(lambda x: x.split('-')[-1])
            final_df['title'] = final_df['title'].apply(lambda x: x.replace("e.g.", ""))
            final_df.rename(columns={'title': 'skills'})
            final_df.drop(columns=['subgroup_id', 'id'], inplace=True)
            final_df.to_csv('../data/processed/subset_df.csv')

            return final_df

    def build_final(self, inventor, patent):
        save_location = f'{self.save_path}/patent_final.csv'
        patent_chunk_list = []
        chunk_list_inventor = []
        chunk_list_patent = []
        df_to_csv = pd.DataFrame()
        try:
            pd.read_csv(save_location, chunksize=1)
            print("Final file exists. Processing further")
            return save_location

        except FileNotFoundError:
            start_time = time.time()
            inventor_location_chunk = pd.read_csv(inventor, chunksize=100000)

            for data_chunk in inventor_location_chunk:
                chunk_list_inventor.append(data_chunk)
            for chunk in range(len(chunk_list_inventor)):
                self.inventor_location = pd.concat([self.inventor_location, chunk_list_inventor[chunk]])

            patent_df = pd.read_csv(patent, chunksize=100000)

            for data_chunk in patent_df:
                chunk_list_patent.append(data_chunk)
            for chunk in range(len(chunk_list_patent)):
                self.patent_location = pd.concat([self.patent_location, chunk_list_patent[chunk]])

            if len(self.inventor_location) == 0 or len(self.patent_location) == 0:
                exit()
            else:
                self.patent_final = pd.merge(self.patent_location, self.inventor_location, how='inner', left_on='id',
                                             right_on='patent_id')

                self.patent_final = self.patent_final.drop(['patent_id', 'type', 'number'], axis=1)
                self.patent_final.rename(columns={'id': 'patent_id'}, inplace=True)
                subset_df = self.subset_patents()
                self.patent_final = pd.merge(self.patent_final, subset_df, how='inner', left_on='id',
                                             right_on='patent_id')
                self.patent_final.drop_duplicates(inplace=True, ignore_index=True)
                self.patent_final.dropna(inplace=True)

                print("Saving dataframe")
                self.patent_final.to_csv(save_location, index=False)
                print(f'Common Records {len(subset_df) - len(self.patent_final)}')
                print(
                    f'Reading and compilation of final data complete and it took {(time.time() - start_time) / 60} minutes')
                return save_location

    def read_data(self, load_file, toy=False):
        print(toy)
        start_time = time.time()
        patent_save_location = f'{self.save_path}/patent.csv'
        inventor_save_location = f'{self.save_path}/inventor.csv'
        subset_df = self.subset_patents()
        to_obj = pd.DataFrame()
        final_save = '../data/processed/patent_all.csv'

        if os.path.isfile(final_save):
            return final_save
        else:
            for item in load_file:
                if item == "inventor":
                    print("Inventor ")
                    chunk_list_inventor = []
                    inventor_df = pd.DataFrame()
                    save_location = f'{self.save_path}/{item}.csv'
                    self.inventor_save_location = save_location
                    if os.path.isfile(inventor_save_location):
                        continue
                    else:
                        print("Here")
                        print("Reading patent_inventor")
                        inventor_chunk = pd.read_csv(self.inventor_path, sep='\t', chunksize=100000)
                        for data_chunk in inventor_chunk:
                            chunk_list_inventor.append(data_chunk)
                        for chunk in range(len(chunk_list_inventor)):
                            self.inventor_location = pd.concat([self.inventor_location, chunk_list_inventor[chunk]])
                        self.inventor_f = self.inventor_location[self.inventor_location['patent_id'].isin(list(subset_df['patent_id']))]
                        print(len(self.inventor_f))

                        # To get names
                        inventor_details = pd.read_csv(self.inventor_details, error_bad_lines=False, sep='\t',
                                                       usecols=['id', 'name_first', 'name_last', 'male_flag'])

                        inventor_final = pd.merge(inventor_details, self.inventor_f, left_on='id', right_on='inventor_id',
                                                  how='inner')
                        inventor_final.drop(['patent_id'], inplace=True, axis=1)
                        inventor_final.rename(columns={'id': 'patent_id'}, inplace=True)
                        print(len(inventor_final))

                        location = pd.read_csv(self.location_path, sep='\t', usecols=['id', 'city', 'state', 'country'])
                        inventor_final_loc = pd.merge(inventor_final, location, left_on='location_id', right_on='id',
                                                      how='left')
                        print(len(inventor_final_loc))
                        inventor_final_loc.drop(['id'], inplace=True, axis=1)
                        inventor_final_loc.dropna(inplace=True)
                        inventor_final_loc.drop_duplicates(inplace=True)
                        inventor_final_loc.reset_index(drop=True)
                        inventor_final_loc.to_csv(inventor_save_location, index=False)

                        print(f'Time taken to process inventor details {(time.time() - start_time) / 60} minutes')

                elif item == "patent":
                    chunk_list_patent = []
                    patent_final = pd.DataFrame()
                    patent_save_location = f'{self.save_path}/{item}.csv'
                    if os.path.isfile(patent_save_location):
                        continue
                    else:
                        print("Started Processing for Patent Table")
                        patent_df = pd.read_csv(self.patent_path, sep='\t', chunksize=100000,
                                                usecols=['id', 'country', 'date', 'title'])
                        for data_chunk in patent_df:
                            chunk_list_patent.append(data_chunk)
                        for chunk in range(len(chunk_list_patent)):
                            self.patent_location = pd.concat([self.patent_location, chunk_list_patent[chunk]]).drop_duplicates()
                        patent_final = self.patent_location[self.patent_location['id'].isin(list(subset_df['patent_id']))]
                        patent_final.rename(columns={'country': 'patent_country'}, inplace=True)
                        print(len(patent_final))
                        patent_final.to_csv(patent_save_location, index=False)
                else:
                    print("Not a valid argument.")
        in_df = pd.read_csv(inventor_save_location)
        pat_df = pd.read_csv(patent_save_location)
        to_obj = pd.merge(in_df, pat_df, left_on='patent_id', right_on='id', how='left')
        to_obj.drop(['id'], axis=1, inplace=True)
        print(len(to_obj))
        to_obj.to_csv(final_save, index=False)
        print(f'Finish Time {(time.time() - start_time) / 60} Minutes ')
        return final_save




    def pack(self):
        # subset_list = self.subset_patents()
        save_location = f'{self.save_path}/patent_final.csv'
        final_df = pd.read_csv(save_location)
        team_dict = {}
        idx = 0
        team_idx = {}
        for _, rows in final_df.iterrows():
            patent_id = rows['patent_id']
            members = rows['inventor_id']
            loc = rows['location_id']
            team_i = author(patent_id, members, loc)
            team_dict[patent_id] = team_i

            idx += 1
        i2c, c2i = Team.build_index_candidates(team_dict.values())
        print(i2c, c2i)
        exit()

    def build_index(self):
        all_inventors = {}
        count = 0
        t = time.time()
        chunk_counter = 0
        mat_df = pd.DataFrame()
        try:

            patent_chunks_list = []
            save_location = f'{self.save_path}/patent_final.csv'
            patent_chunks = pd.read_csv(save_location, chunksize=100000)

            for data_chunk in patent_chunks:
                patent_chunks_list.append(data_chunk)

            for each_chunk in patent_chunks_list:
                chunk_counter += 1
                inv = each_chunk[['patent_id', 'inventor_id', 'location_id']]
                loc = f'{self.save_path}/indexed_inventor.csv'
                # inv.to_csv(loc, index=False)
                n_ids = len(inv['inventor_id'])
                df_idx = inv['patent_id']
                for _, rows in inv.iterrows():
                    count += 1
                    patent_id = rows['patent_id']
                    inventor = rows['inventor_id']
                    loc = rows['location_id']
                    teams = author(patent_id, inventor, loc)
                    if patent_id not in all_inventors.keys():
                        all_inventors[patent_id] = [inventor]
                    elif type(all_inventors[patent_id]) == list:
                        all_inventors[patent_id].append(inventor)
                    else:
                        all_inventors[patent_id] = [all_inventors[patent_id], inventor]

                    X = np.zeros((len(all_inventors.keys()), n_ids))
                    cmx = 0
                    rowx = 0
                    idx = {}
                    for key, value in all_inventors.items():
                        for val_id in value:
                            X[rowx, cmx] = 1
                            if val_id not in idx.keys():
                                idx[val_id] = [rowx]
                            elif type(idx[val_id]) == list:
                                idx[val_id].append(rowx)
                            else:
                                idx[val_id] = [idx[val_id], rowx]
                            cmx += 1
                        rowx += 1

                    temp_df = pd.DataFrame(0, columns=list(set(idx.keys())), index=df_idx)

                    for key, value in idx.items():
                        temp_df.iloc[value, temp_df.columns.get_loc(key)] = 1
                    mat_df = pd.concat([mat_df, temp_df]).drop_duplicates()
                    print(f'Processed  Number of Rows {count}')
                print(f'Remaining Number of Chunks {len(patent_chunks_list) - chunk_counter}')
            fi_mat = sp.csr_matrix(mat_df.values)

            with open(f'{self.save_path}/spmat_inventor.pkl', "wb") as outfile:
                pickle.dump(fi_mat, outfile)

            with open(f'{self.save_path}/spdf_inventor.pkl', "wb") as outfile:
                pickle.dump(mat_df, outfile)

            print(f'Total Time Taken to build and store Sparse Matrix {(time.time() - t) / 60} minutes')

        except FileNotFoundError:
            print('Here')

