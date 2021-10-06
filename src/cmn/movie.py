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
from cmn.castncrew import CastnCrew


class Movie(Team):

    def __init__(self, id, members, p_title, o_title, release, end, runtime, genres, members_details):
        super().__init__(id, members, None, release)
        self.p_title = p_title
        self.o_title = o_title
        self.release = release
        self.end = end
        self.runtime = runtime
        self.genres = genres
        self.members_details = members_details
        self.skills = self.set_skills()

        for i, member in enumerate(self.members):
            member.teams.add(self.id)
            member.skills.union(set(self.skills))
            member.role.add(self.members_details[i])

    def set_skills(self):
        return set(self.genres.split(','))
    
    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        try:
            return super(Movie, Movie).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print("Pickles not found! Reading raw data ...")
            #in imdb, titles represent movies and name represent crew members

            title_basics = pd.read_csv(datapath, sep='\t', header=0, na_values='\\N').sort_values(by=['tconst'])#title.basics.tsv
            title_basics = title_basics[title_basics['titleType'].isin(['movie', ''])]
            title_principals = pd.read_csv(datapath.replace('title.basics', 'title.principals'), sep='\t', header=0, na_values='\\N')#movie-crew association for top-10 cast
            name_basics = pd.read_csv(datapath.replace('title.basics', 'name.basics'), sep='\t', header=0, na_values='\\N')#name.basics.tsv

            movies_crewids = pd.merge(title_basics, title_principals, on='tconst', how='inner', copy=False)
            movies_crewids_crew = pd.merge(movies_crewids, name_basics, on='nconst', how='inner', copy=False)

            movies_crewids_crew.dropna(subset=['genres'], inplace=True)
            movies_crewids_crew = movies_crewids_crew.append(pd.Series(), ignore_index=True)

            teams = {}
            candidates = {}
            n_row = 0
            current = None
            members = []; members_details = []
            for index, movie_crew in movies_crewids_crew.iterrows():
                try:
                    if pd.isnull(new := movie_crew['tconst']): break
                    if current != new:
                        team = Movie(movie_crew['tconst'].replace('tt', ''),
                                     [],
                                     movie_crew['primaryTitle'],
                                     movie_crew['originalTitle'],
                                     int(movie_crew['startYear']) if not pd.isnull(movie_crew['startYear']) else None,
                                     int(movie_crew['endYear']) if not pd.isnull(movie_crew['endYear']) else None,
                                     movie_crew['runtimeMinutes'],
                                     movie_crew['genres'],
                                     [])
                        current = new
                        teams[team.id] = team

                    member_id = movie_crew['nconst'].replace('nm', '')
                    member_name = movie_crew['primaryName'].replace(" ", "_")
                    if (idname := f'{member_id}_{member_name}') not in candidates:
                        candidates[idname] = CastnCrew(movie_crew['nconst'].replace('nm', ''),
                                                       movie_crew['primaryName'].replace(' ', '_'),
                                                       int(movie_crew['birthYear']) if not pd.isnull(movie_crew['birthYear']) else None,
                                                       int(movie_crew['deathYear']) if not pd.isnull(movie_crew['deathYear']) else None,
                                                       movie_crew['primaryProfession'],
                                                       movie_crew['knownForTitles'],
                                                       None)
                    team.members.append(candidates[idname])
                    team.members_details.append((movie_crew['category'], movie_crew['job'], movie_crew['characters']))



                    if (n_row := n_row + 1) % 10000 == 0: print(f"{n_row} instances have been loaded, and {time() - st} seconds has passed.")

                except Exception as e:
                    raise e

            return super(Movie, Movie).read_data(teams, output, filter, settings)
