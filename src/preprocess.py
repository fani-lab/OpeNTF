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



class Assignee:
    def __init__(self, assignee_path, inventor_path, location_path, save_path, patent_path):
        self.assignee_path = assignee_path
        self.inventor_path = inventor_path
        self.location_path = location_path
        self.save_path = save_path
        self.assignee_save_location = ""
        self.inventor_save_location = ""
        #self.patent_save_location = ""
        self.assignee_location = pd.DataFrame()
        self.inventor_location = pd.DataFrame()
        self.patent_location = pd.DataFrame()
        self.patent_path = patent_path
        self.cpc_path = '../data/raw/cpc_current.tsv'
        self.subgroup_path = '../data/raw/cpc_subgroup.tsv'
        self.location = pd.read_csv(self.location_path, sep='\t', usecols=['id', 'city', 'state', 'country'])
        self.patent_final = pd.DataFrame()
        self.dblp_path = '../data/raw/dblp.v12.json'

    def subset_patents(self):
        ner_list = []
        set_x = set()
        cpc_list = []
        start_time = time.time()
        cpc_chunks = pd.read_csv(self.cpc_path, sep='\t', chunksize=100000, usecols=['patent_id', 'subgroup_id'])
        df_concat = pd.DataFrame()
        for chunk in cpc_chunks:
            df_concat = pd.concat([df_concat, chunk[chunk['subgroup_id'].str.contains('^Y10S706', regex=True)]])
        df_concat.reset_index(drop=True)
        subgroup_df = pd.read_csv(self.subgroup_path, sep='\t')
        final_df = pd.merge(df_concat[['patent_id', 'subgroup_id']], subgroup_df, left_on='subgroup_id', right_on='id')
        for val in final_df['title']:
            set_x.add(val.split('-')[-1])
        new_set = {x.replace("e.g.", "") for x in set_x}

        for ner in new_set:
            ner_list.extend(x.replace(" ", "") for x in ner.split(','))
        print(ner_list)
        print(f'Time Taken to create sublist {(time.time() - start_time)/60} minutes')
        return ner_list
        #return final_df['patent_id'].to_list()

    def build_final(self, assignee, inventor, patent):
        save_location = f'{self.save_path}/patent_final.csv'
        patent_chunk_list = []
        chunk_list_assignee = []
        chunk_list_inventor = []
        chunk_list_patent = []
        df_to_csv = pd.DataFrame()
        try:
            pd.read_csv(save_location, chunksize=100000)
            print("Final file exists. Processing further")
            return save_location

        except FileNotFoundError:
            print("Building Final Dataframe...")
            start_time = time.time()
            # assignee_location_chunk = pd.read_csv(assignee, chunksize=100000)
            # print("File exists. Reading the file.")
            # for data_chunk in assignee_location_chunk:
            #     chunk_list_assignee.append(data_chunk)
            # for chunk in range(len(chunk_list_assignee)):
            #     self.assignee_location = pd.concat([self.assignee_location, chunk_list_assignee[chunk]])

            inventor_location_chunk = pd.read_csv(inventor, chunksize=100000)
            print("File exists. Reading the file.")
            for data_chunk in inventor_location_chunk:
                chunk_list_inventor.append(data_chunk)
            for chunk in range(len(chunk_list_inventor)):
                self.inventor_location = pd.concat([self.inventor_location, chunk_list_inventor[chunk]])

            patent_df = pd.read_csv(patent, chunksize=100000)
            print("Patent File Already exists. Reading the file.")
            for data_chunk in patent_df:
                chunk_list_patent.append(data_chunk)
            for chunk in range(len(chunk_list_patent)):
                self.patent_location = pd.concat([self.patent_location, chunk_list_patent[chunk]])

            if len(self.inventor_location) == 0 or len(self.patent_location) == 0:
                print("Data not Complete. Please consider running read method again.")
            else:
                print("Any moment now")
                self.patent_final = pd.merge(self.patent_location, self.inventor_location, how='inner', left_on='id',
                                             right_on='patent_id')

                self.patent_final = self.patent_final.drop(['patent_id', 'type', 'number'], axis=1)
                self.patent_final.rename(columns={'id': 'patent_id'}, inplace=True)
                subset_list = self.subset_patents()
                print()
                self.patent_final = self.patent_final[self.patent_final['patent_id'].isin(list(set(subset_list)))]
                self.patent_final.drop_duplicates(inplace=True, ignore_index=True)
                self.patent_final.dropna(inplace=True)
                # for id_sub in subset_list:
                #     temp_df = self.patent_final.loc[self.patent_final['patent_id'] == id_sub]
                #     df_to_csv = pd.concat([df_to_csv, temp_df])
                #     print(df_to_csv)

                print("Saving dataframe")
                self.patent_final.to_csv(save_location, index=False)
                print(f'Common Records {len(subset_list) - len(self.patent_final)}')
                print(f'Reading and compilation of final data complete and it took {(time.time() - start_time)/60} minutes')
                return save_location

    def read_data(self, load_file):
        start_time = time.time()
        patent_save_location = f'{self.save_path}/patent.csv'
        assignee_save_location = f'{self.save_path}/assignee.csv'
        inventor_save_location = f'{self.save_path}/inventor.csv'
        for item in load_file:
            if item == "assignee":
                chunk_list_assignee = []

                save_location = f'{self.save_path}/{item}.csv'
                self.assignee_save_location = save_location

                try:
                    pd.read_csv(assignee_save_location, chunksize=100000)
                    break
                    # for data_chunk in assignee_location_chunk:
                    #     chunk_list_assignee.append(data_chunk)
                    # for chunk in range(len(chunk_list_assignee)):
                    #     self.assignee_location = pd.concat([self.assignee_location, chunk_list_assignee[chunk]])
                    #     print(len(self.assignee_location))

                except FileNotFoundError:
                    print("File Not Found!. Running Preprocessing for patent_assignee table")
                    assignee_df = pd.read_csv(self.assignee_path, sep='\t', chunksize=100000)
                    for data_chunk in assignee_df:
                        chunk_list_assignee.append(data_chunk)
                    for each_chunk in range(len(chunk_list_assignee)):
                        s1 = pd.merge(chunk_list_assignee[each_chunk], self.location, how='inner', left_on='location_id',
                                      right_on='id')

                        print(f'Length of common ids for chunk {each_chunk} is {len(s1)}')
                        self.assignee_location = pd.concat([self.assignee_location, s1]).drop_duplicates()

                    self.assignee_location.drop(['id'], inplace=True, axis=1)
                    self.assignee_location.dropna(inplace=True)
                    self.assignee_location.drop_duplicates(inplace=True)

                    self.assignee_location.to_csv(assignee_save_location, index=False)
                    print(f'Time taken to process this file{time.time() - start_time}')

            elif item == "inventor":
                chunk_list_inventor = []

                save_location = f'{self.save_path}/{item}.csv'

                self.inventor_save_location = save_location
                try:
                    pd.read_csv(inventor_save_location, chunksize=100000)
                    # inventor_location_chunk = pd.read_csv(save_location, chunksize=100000)
                    # print("File Already exists. Reading the file.")
                    # for data_chunk in inventor_location_chunk:
                    #     chunk_list_inventor.append(data_chunk)
                    # for chunk in range(len(chunk_list_inventor)):
                    #     self.inventor_location = pd.concat([self.inventor_location, chunk_list_inventor[chunk]])

                except FileNotFoundError:
                    inventor_df = pd.read_csv(self.inventor_path, sep='\t', chunksize=100000)
                    print("File Not Found!. Running Preprocessing for patent_inventor table")
                    for data_chunk in inventor_df:
                        chunk_list_inventor.append(data_chunk)
                    for each_chunk in range(len(chunk_list_inventor)):
                        s1 = pd.merge(chunk_list_inventor[each_chunk], self.location, how='inner', left_on='location_id',
                                      right_on='id')
                        print(f'Length of common ids for chunk {each_chunk} is {len(s1)}')

                        self.inventor_location = pd.concat([self.inventor_location, s1]).drop_duplicates()
                    self.inventor_location.drop(['id'], inplace=True, axis=1)
                    self.inventor_location.dropna(inplace=True)
                    self.inventor_location.drop_duplicates(inplace=True)
                    self.inventor_location.reset_index(drop=True)
                    self.inventor_location.to_csv(inventor_save_location, index=False)
                    print(f'Time taken to process this file {(time.time() - start_time)/60} minutes')

            elif item == "patent":
                chunk_list_patent = []

                self.patent_save_location = f'{self.save_path}/{item}.csv'
                #$print(save_location)
                #self.patent_save_location = save_location
                try:
                    pd.read_csv(patent_save_location, chunksize=100000)
                    # print("Patent File Already exists. Reading the file.")
                    # for data_chunk in patent_df:
                    #     chunk_list_patent.append(data_chunk)
                    # for chunk in range(len(chunk_list_patent)):
                    #     self.patent_location = pd.concat([self.patent_location, chunk_list_patent[chunk]])
                    #     print(len(self.patent_location))

                except FileNotFoundError:
                    print("Started Processing for Patent Table")
                    patent_df = pd.read_csv(self.patent_path, sep='\t', chunksize=100000,
                                            usecols=['id', 'number', 'country', 'date', 'title'])
                    for data_chunk in patent_df:
                        chunk_list_patent.append(data_chunk)
                    for chunk in range(len(chunk_list_patent)):
                        print(f'Processed Chunk Number {chunk}')
                        self.patent_location = pd.concat([self.patent_location, chunk_list_patent[chunk]]).drop_duplicates()

                    self.patent_location.to_csv(patent_save_location, index=False)
            else:
                print("Not a valid argument.")

        final_save = self.build_final(assignee_save_location, inventor_save_location, patent_save_location)
        return final_save

    def extract_abstract(self):
        abstract_dict = {}
        #subset_patents = self.subset_patents()
        save_location = f'{self.save_path}/patent_abstract.csv'
        print('Subset Fetch Completed')
        abstract_df = pd.DataFrame()
        skills_pre = gs.Skills()
        s2i = {}
        try:
            print("Here")

            abstract_chunks = pd.read_csv(f'{self.save_path}/patent_abstract.csv', chunksize=100000)
            for each_chunk in abstract_chunks:
                abstract_df = pd.concat([abstract_df, each_chunk])
            abstract_dict = abstract_df.set_index('id').T.to_dict('list')
            skills_set = skills_pre.text_preprocess(abstract_dict)
            skills_cit = skills_pre.get_skills(self.dblp_path)
            for key, value in skills_set.items():
                temp_set = skills_cit.intersection(value)
                if len(temp_set) != 0:
                    s2i[key] = list(temp_set)
                else:
                    continue

            #print(s2i)
            exit()
            #print(abstract_dict)
            # One Hot encoding

        except FileNotFoundError:

            save_location = f'{self.save_path}/patent_abstract.csv'
            abstract_df = pd.DataFrame()
            abstract_subset = pd.DataFrame()
            chunk_list_abstract = []
            patent_df = pd.read_csv(self.patent_path,  sep='\t', chunksize=100000, usecols=['id', 'abstract'])
            for data_chunk in patent_df:
                chunk_list_abstract.append(data_chunk)

            for chunk in range((len(chunk_list_abstract))):
                abstract_df = pd.concat([abstract_df, chunk_list_abstract[chunk]]).drop_duplicates()

            abstract_subset = abstract_df[abstract_df['id'].isin(list(set(subset_patents)))]
            abstract_dict = abstract_subset.set_index('id').T.to_dict('list')

            skills_pre.text_preprocess(abstract_dict)
            abstract_subset.to_csv(save_location, index=False)

    def pack(self):
        subset_list = self.subset_patents()
        save_location = f'{self.save_path}/patent_final.csv'
        final_df = pd.read_csv(save_location)
        team_dict = {}

        for _, rows in final_df.iterrows():
            patent_id = rows['patent_id']
            members = rows['inventor_id']
            loc = rows['location_id']

            team_i = author(patent_id, members, loc)
            team_dict[patent_id] = team_i

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
                #inv.to_csv(loc, index=False)
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

            print(f'Total Time Taken to build and store Sparse Matrix {(time.time() - t)/60} minutes')

        except FileNotFoundError:
            print('Here')

