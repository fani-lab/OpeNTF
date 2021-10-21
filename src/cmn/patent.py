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


class Patent:
    def __init__(self, assignee_path, inventor_path, location_path, save_path, patent_path):
        self.inventor_path = inventor_path
        self.location_path = location_path
        self.save_path = save_path
        self.inventor_save_location = ""
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
            final_df.rename(columns={'title': 'skills'}, inplace=True)
            final_df.drop(columns=['subgroup_id', 'id'], inplace=True)
            final_df = final_df.sort_values(by=['patent_id'])
            final_df.to_csv('../data/processed/subset_df.csv', index=False)
            return final_df

    def read_data(self, load_file, toy=False):

        start_time = time.time()
        patent_save_location = f'{self.save_path}/patent.csv'
        inventor_save_location = f'{self.save_path}/inventor.csv'
        subset_df = self.subset_patents()
        to_obj = pd.DataFrame()
        final_save = '../data/processed/patent_processed.csv'
        toy_save = '../data/raw/toy.patent'
        chunk_list_patent = []

        if os.path.isfile(final_save):
            if toy:
                patent_df = pd.read_csv(final_save)
                Patent.toy(patent_df, toy_save)
            patent_df = pd.read_csv(final_save)
            return final_save
        else:
            chunk_list_inventor = []
            inventor_df = pd.DataFrame()
            print("Here")
            print("Reading patent_inventor")
            inventor_chunk = pd.read_csv(self.inventor_path, sep='\t', chunksize=100000)
            for data_chunk in inventor_chunk:
                chunk_list_inventor.append(data_chunk)
            for chunk in range(len(chunk_list_inventor)):
                self.inventor_location = pd.concat([self.inventor_location, chunk_list_inventor[chunk]])
            inventor_f = pd.merge(self.inventor_location, subset_df, left_on='patent_id', right_on='patent_id',
                                  how='inner')

            # To get names
            inventor_details = pd.read_csv(self.inventor_details, error_bad_lines=False, sep='\t',
                                           usecols=['id', 'name_first', 'name_last', 'male_flag'])
            inventor_details['full_name'] = inventor_details['name_first'] + '_' + inventor_details['name_last']
            inventor_details.drop(columns=['name_first', 'name_last'], inplace=True)

            inventor_final = pd.merge(inventor_f, inventor_details, left_on='inventor_id', right_on='id',
                                      how='inner')
            inventor_final.drop(['id'], inplace=True, axis=1)

            print(len(inventor_final))
            location = pd.read_csv(self.location_path, sep='\t', usecols=['id', 'city', 'state', 'country'])
            inventor_final_loc = pd.merge(inventor_final, location, left_on='location_id', right_on='id',
                                          how='inner')

            inventor_final_loc.drop(columns=['id'], inplace=True, axis=1)

            patent_df = pd.read_csv(self.patent_path, sep='\t', chunksize=100000,
                                    usecols=['id', 'date', 'title'])
            for data_chunk in patent_df:
                chunk_list_patent.append(data_chunk)
            for chunk in range(len(chunk_list_patent)):
                self.patent_location = pd.concat(
                    [self.patent_location, chunk_list_patent[chunk]]).drop_duplicates()
            patent_final = pd.merge(inventor_final_loc, self.patent_location, left_on='patent_id',
                                    right_on='id', how='inner')
            patent_final.drop(columns=['id'], inplace=True)
            if toy:
                Patent.toy(patent_final, toy_save)
            patent_final.to_csv(final_save, index=False)

            print(f'Time taken to process inventor details {(time.time() - start_time) / 60} minutes')

        return final_save

    @staticmethod
    def toy(patent_df, toy_save):
        toy_df = pd.DataFrame()
        toy_df = patent_df[patent_df['country'].isin(['US'])][:4]
        toy_df = pd.concat([toy_df, patent_df[patent_df['country'].isin(['GB'])][:3]])
        toy_df = pd.concat([toy_df, patent_df[patent_df['country'].isin(['IN'])][:3]])
        toy_inventor = toy_df[['patent_id', 'inventor_id', 'full_name', 'male_flag']]
        toy_patent_inventor = toy_df[['patent_id', 'inventor_id', 'skills', 'date', 'title']]
        toy_location = toy_df[['patent_id', 'location_id', 'city', 'state', 'country']]
        toy_df.to_csv(f'{toy_save}/toy.patent.csv', index=False)
        toy_inventor.to_csv(f'{toy_save}/toy.inventor.csv', index=False)
        toy_patent_inventor.to_csv(f'{toy_save}/toy.patent_inventor.csv', index=False)
        toy_location.to_csv(f'{toy_save}/toy.location.csv', index=False)

