import pickle
import pandas as pd
import os
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, teams_file, teamsvecs_file, indexes_file):
        with open(teams_file, 'rb') as infile: self.teams = pickle.load(infile)
        with open(teamsvecs_file, 'rb') as infile: self.teamsvecs = pickle.load(infile)
        with open(indexes_file, 'rb') as infile: self.indexes = pickle.load(infile)
        self.train_dict = {}; self.test_dict = {}
        self.save_dict = {}
        self.data_df = pd.DataFrame()
        # self.generate_data()
        DataLoader.create_misc()
        # DataLoader.create_kg()
        # DataLoader.print_info()

    def generate_data(self):
        try:
            print('Reading data file...')
            with open('../data/preprocessed/uspt/toy.patent.tsv/data.txt') as f:
                lines = f.readlines()
                for l in lines:
                    self.train_dict[int(l.split(' ')[0])] = list(map(int, l.split(' ')[1:-1]))
            self.split_data()
        except FileNotFoundError:
            mid_dict = dict()
            print('File Not Found. Generating..')
            # print(self.teamsvecs['loc'])
            del self.teamsvecs['skill']
            self.teamsvecs['skill'] = self.teamsvecs['loc']
            del self.teamsvecs['loc']
            # print(self.teamsvecs)
            for team in self.teams.values():
                mem_list = []
                loc_list = []
                for candidate in team.members:
                    idname = f'{candidate.id}_{candidate.name}'
                    mem_list.append(self.indexes['c2i'][idname])
                for loc in team.members_details:
                    loc_list.append(self.indexes['l2i'][loc[2]])

                for k in range(len(mem_list)):
                    if mem_list[k] in list(mid_dict.keys()):
                        mid_dict[mem_list[k]].append(loc_list[k])
                    else:
                        mid_dict.update({mem_list[k]: [loc_list[k]]})
            for key, value in mid_dict.items():
                # print(key, value)
                self.save_dict[key] = ''
                for item in value:
                    # print(item)
                    self.save_dict[key] = self.save_dict[key] + str(item) + ' '
            self.write_data()
            self.split_data()

    def write_data(self):
        print('Writing data into file data.txt')
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'w') as f:
            for key, value in self.save_dict.items():
                # tmp = str(str(key) + " " + str(value) + '\n')
                f.write('%s %s\n' % (key, value))
                # f.write(tmp)

    def split_data(self):
        try:
            print('Reading train and test')
            with open('../data/preprocessed/uspt/toy.patent.tsv/train.txt') as f:
                lines = f.readlines()
                for l in lines:
                    self.train_dict[int(l.split(' ')[0])] = list(map(int, l.split(' ')[1:-1]))
            with open('../data/preprocessed/uspt/toy.patent.tsv/test.txt') as f:
                lines = f.readlines()
                for l in lines:
                    self.test_dict[int(l.split(' ')[0])] = list(map(int, l.split(' ')[1:-1]))
            print('Read Complete.')
        except FileNotFoundError:
            print('Train, Test File does not exist. Generating...')
            self.data_df = pd.DataFrame(self.save_dict.items())
            train, test = train_test_split(self.data_df, test_size=0.2, random_state=2019)
            self.train_dict = dict(zip(train.iloc[:, 0], train.iloc[:, 1]))
            self.test_dict = dict(zip(test.iloc[:, 0], test.iloc[:, 1]))
            train.to_csv('../data/preprocessed/uspt/toy.patent.tsv/train.txt', header=None, index=None, sep=' ', mode='a')
            test.to_csv('../data/preprocessed/uspt/toy.patent.tsv/test.txt', header=None, index=None, sep=' ', mode='a')

    @staticmethod
    def create_kg():
        with open('../data/preprocessed/uspt/toy.patent.tsv/data.txt') as f:
            lines = f.readlines()
            kg_df = pd.DataFrame(columns=['h', 'r', 't'])
            for l in lines:
                # tmp = l.strip()
                x = l.split(' ')
                id = x[0]
                for point in x[1:-1]:
                    df = pd.DataFrame(columns=['h', 'r', 't'])
                    # val_dict = {id: point}
                    df.loc[0] = [id, 0, point]
                    # print(df)
                    # exit()
                    kg_df = pd.concat([kg_df, df])
                    # print(kg_df)
            kg_df.to_csv('../data/preprocessed/uspt/toy.patent.tsv/kg_final.txt', header=None, index=None, sep=' ', mode='a')
            print(kg_df)

    @staticmethod
    def create_misc():
        user_list = pd.DataFrame(columns=['org_id', 'remap_id'])
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'r') as f:
            lines = f.readlines()
            user_set = []
            for ix, l in enumerate(lines):
                x = l.split(' ')
                id = x[0]
                if id not in user_set:
                    user_set.append(id)
                # df = pd.DataFrame(columns=['org_id', 'remap_id'])
                    user_list.loc[ix] = [id, id]
        user_list.to_csv('../data/preprocessed/uspt/toy.patent.tsv/user_list.txt', index=None, sep=' ', mode='w')

        item_list = pd.DataFrame(columns=['org_id', 'remap_id', 'freebase_id'])
        entity_list = pd.DataFrame(columns=['org_id', 'remap_id'])
        item_set = []
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'r') as f:
            lines = f.readlines()
            for ix, l in enumerate(lines):
                x = l.split(' ')
                for item in x[1:-1]:
                    if item not in item_set:
                        item_list.loc[ix] = [item, item, item]
                        item_set.append(item)
        item_list.reset_index(inplace=True, drop=True)
        item_list.to_csv('../data/preprocessed/uspt/toy.patent.tsv/item_list.txt', index=None, sep=' ', mode='w')

        entity_list['org_id'] = item_list['freebase_id']
        entity_list['remap_id'] = item_list['remap_id']
        entity_list.to_csv('../data/preprocessed/uspt/toy.patent.tsv/entity_list.txt', index=None, sep=' ', mode='w')

        relation_list = pd.DataFrame(columns=['org_id', 'remap_id'])
        relation_list.loc[0] = ['BelongsFrom', 0]

        relation_list.to_csv('../data/preprocessed/uspt/toy.patent.tsv/relation_list.txt', index=None, sep=' ', mode='w')

    @staticmethod
    def print_info():
        with open("../data/preprocessed/uspt/toy.patent.tsv/data.txt", 'r') as f:
            nice = []
            lines = f.readlines()
            # print('Here')
            # print(lines)
            for l in lines:
                nice.append(l)
            train, test = train_test_split(nice, test_size=0.2, random_state=2019)
            # print(train)
            for item in train:
                with open('../data/preprocessed/uspt/toy.patent.tsv/train_new.txt', 'a') as fi:
                    fi.write(item)
            for item in test:
                with open('../data/preprocessed/uspt/toy.patent.tsv/test_new.txt', 'a') as fo:
                    fo.write(item)
            print(nice)


