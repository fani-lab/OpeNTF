import pandas as pd
import traceback
import pickle
from time import time
import os
from collections import Counter
from os import listdir, write
from csv import DictReader
from csv import reader
from cmn.team import Team
from cmn.member import Member


class Movie(Team):
    
    #constructor
    def __init__(self,tconst,members,primaryTitle,originalTitle, startYear, endYear, genres):
        super().__init__(tconst, members,None, startYear)
        self.tconst = tconst
        self.primaryTitle = primaryTitle
        self.originalTitle = originalTitle
        self.startYear = startYear
        self.endYear = endYear
        self.genres = genres
        self.skills = self.set_skills()

        for member in self.members:
            member.teams.add(self.id)
            member.skills.union(set(self.skills))

    def get_tconst(self):
        return self.tconst
    
    def get_startYear(self):
        return self.startYear
    
    #Filling the skills for the every movie
    def set_skills(self):
        skills = set()
        for skill in self.genres.split(','):
            skills.add(skill)
        # print(skills)
        return skills
    
    def get_skills(self):
        return super().get_skills()

    @staticmethod
    def read_data(datapath,output,index,filter,settings):
        try:    
            st = time()
            print("Loading indexes pickle...")
            with open(f'{output}/movie_indexes.pkl', 'rb') as infile: indexes = pickle.load(infile)
            print(f"It took {time() - st} seconds to load from the pickles.")
            teams = None
            if not index:
                st = time()
                print("Loading teams pickle...")
                with open(f'{output}/movie_teams.pkl', 'rb') as tfile: teams = pickle.load(tfile)
                print(f"It took {time() - st} seconds to load from the pickles.")
            return indexes, teams

        except (FileNotFoundError, EOFError) as e:
            print("Pickles not found! Reading raw data ...")
            #Initializing path for csv files
            basic_csv = f'{datapath}'
            principal_csv = ("/").join(datapath.split('/')[:-1]) + '/title_principal.csv'

            teams = {}
            candidates = {}
            counter = 0
            with open(basic_csv, encoding='utf-8') as file1,open(principal_csv,encoding='utf-8') as file2:
                #Reading csv files using DictReader
                csv_basic = DictReader(file1)
                csv_principle = DictReader(file2)

                start_time = time()
                nconst_mem = None
                nconst_category = None
                tconst = ""; nconst = ""; primarytitle = ""; originaltitle = "";
                startYear = ""; endYear = ""; genres = ""

                #Reading line by line from basic_csv file
                for row in csv_basic:
                    try:
                        #Retriving required data
                        tconst = row['tconst']
                        originalTitle = row['originalTitle']
                        startYear = row['startYear']
                        endYear = row['endYear']
                        primarytitle = row['primaryTitle']
                        # genres = row['genres']

                        #Each movie should have skills (genres)
                        try: genres = row['genres']
                        except: print(f'Warning! No genre for team id={tconst}. Bypassed!'); continue

                        #Reading line by line 2nd file and retriving members
                        members = []
                        for row1 in csv_principle:
                            try: member = row1['nconst']
                            except: print(f'Warning! No actors for team id={id}. Bypassed!'); continue #Each movie must have members

                            tconst2 = row1['tconst']

                            #Condition to append last read member with same id
                            if nconst_mem is not None and nconst_category is not None:
                                idname = f'{nconst_mem}_{nconst_category}'
                                if idname not in candidates:
                                    member = Member(nconst_mem,nconst_category)
                                    candidates[idname] = member
                                members.append(candidates[idname])
                                nconst_mem = None
                                nconst_category = None

                            #Condition to check if id(tconst) of both movies are same and than add members
                            if tconst2 == tconst: 
                                idname = f"{row1['nconst']}_{row1['category']}"
                                if idname not in candidates:
                                    member = Member(row1['nconst'],row1['category'])
                                    candidates[idname] = member
                                members.append(candidates[idname])
                                continue
                            else:
                                nconst_mem = row1['nconst']
                                nconst_category = row1['category']
                                break

                        id = int(tconst.split('tt')[1])
                        # print(id)
                        team = Movie(id,members,primarytitle, originalTitle, startYear, endYear,genres)
                        teams[team.tconst] = team

                        counter += 1
                        if counter % 10000 == 0:
                            print(f"{counter} instances have been loaded, and {time() - start_time} seconds has passed.")
                    except:  
                        print(f'ERROR: There has been error in loading line `{row}`!\n{traceback.format_exc()}')
                        continue

                i2c, c2i = Team.build_index_candidates(teams.values())
                i2s, s2i = Team.build_index_skills(teams.values())
                i2t, t2i = Team.build_index_teams(teams.values())
                st = time()
                with open(f'{output}/movie_teams.pkl', "wb") as outfile: pickle.dump(teams, outfile)
                with open(f'{output}/movie_indexes.pkl', "wb") as outfile: pickle.dump((i2c, c2i, i2s, s2i, i2t, t2i), outfile)
                with open(f'{output}/movie_candidates.pkl', "wb") as outfile: pickle.dump(candidates, outfile)
                print(f"It took {time() - st} seconds to pickle the data")

                return super(Movie, Movie).read_data(teams, output, filter, settings)

    def load_data(filename):
        file = open(filename, 'r')

    def get_stats(teams,output):
        try:
            with open(output, 'rb') as infile:
                print("Loading the stat pickle...")
                return pickle.load(infile)

        except FileNotFoundError:
            print("File not found! Generating stats ...")



