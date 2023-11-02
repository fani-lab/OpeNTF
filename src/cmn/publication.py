import json
from tqdm import  tqdm
import traceback
import pickle
from time import time
import os


import numpy as np
from cmn.author import Author
from cmn.team import Team

class Publication(Team):
    def __init__(self, id, authors, title, datetime, doc_type, venue, references, fos, keywords):
        super().__init__(id, authors, None, datetime)
        self.title = title
        self.doc_type = doc_type
        self.venue = venue
        self.references = references
        self.fos = fos
        self.keywords = keywords
        self.skills = self.set_skills()

        for author in self.members:
            author.teams.add(self.id)
            author.skills.update(set(self.skills))

    # Fill the fields attribute with non-zero weight from FOS
    def set_skills(self):
        skills = set()
        for skill in self.fos:
            #if skill["w"] != 0.0:
            skills.add(skill["name"].replace(" ", "_"))
        # Extend the fields with keywords
        # if len(self.keywords):
        #     skills.union(set([keyword.replace(" ", "_") for keyword in self.keywords]))
        return skills

    def get_skills(self):
        return self.skills

    def get_year(self):
        return self.year

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        try:
            # at first try to load the existing data (if any) with the default super.load_data() method
            return super(Publication, Publication).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            # otherwise we read it using the custom read_data() function for each domain class
            print(f"Pickles not found! Reading raw data from {datapath} (progress in bytes) ...")
            teams = {}; candidates = {}

            # if there is no file, load it from scratch
            with tqdm(total=os.path.getsize(datapath)) as pbar, open(datapath, "r", encoding='utf-8') as jf:
                for line in jf:
                    try:
                        if not line: break
                        pbar.update(len(line))
                        jsonline = json.loads(line.lower().lstrip(","))
                        id = jsonline['id']
                        title = jsonline['title']
                        year = jsonline['year']
                        type = jsonline['doc_type']
                        # venue itself is a dict with keys 'raw', 'id' and 'type'
                        # e.g : {'raw': 'international conference on human-computer interaction', 'id': 1127419992, 'type': 'c'}
                        venue = jsonline['venue'] if 'venue' in jsonline.keys() else None
                        # references = [2005687710, 2018037215]
                        references = jsonline['references'] if 'references' in jsonline.keys() else []
                        keywords = jsonline['keywords'] if 'keywords' in jsonline.keys() else []

                        # a team must have skills and members, otherwise skip this paper and continue to the next iteration
                        # fos = a list of dictionaries with keys 'name', 'w'
                        # fos = [{'name': 'deep learning', 'w': 0.45139}, {'name': 'image captioning', 'w': 0.3241}]
                        try: fos = jsonline['fos']# an array of (name, w), w shows a weight. Not sorted! Can be used later!
                        except: continue  #publication must have fos (skills)
                        try: authors = jsonline['authors']
                        except: continue #publication must have authors (members)

                        members = []
                        for author in authors:
                            member_id = author['id']
                            member_name = author['name'].replace(" ", "_")
                            member_org = author['org'].replace(" ", "_") if 'org' in author else ""
                            if (idname := f'{member_id}_{member_name}') not in candidates:
                                # each Author object is stored in the candidates dict against a distinct idname
                                candidates[idname] = Author(member_id, member_name, member_org)
                            members.append(candidates[idname])
                        # declare the team based on the data collected for a publication
                        # in the json file, one line represents the whole team
                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team
                        # this line somehow miss the heirarchy of settings' keys : data > domain > dblp > nrow
                        if 'nrow' in settings['domain']['dblp'].keys() and len(teams) > settings['domain']['dblp']['nrow']: break
                    except json.JSONDecodeError as e:  # ideally should happen only for the last line ']'
                        print(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                        continue
                    except Exception as e:
                        raise e
            # if there is no previous data to load, after this step in the child class called publication,
            # the read_data() of the super class (Team) will be called to create the indexes and the teams
            # and finally store them in the pickle format
            # as the pickle and index formation is the same for every team child class
            # the read_data() should be called by every child class of Team after performing necessary data fetching and preps
            # from the raw files
            return super(Publication, Publication).read_data(teams, output, filter, settings)
        except Exception as e:
            raise e

    # @classmethod
    # def get_stats(cls, teamsvecs, output, plot=False):
        # return super(Publication, cls).get_stats(teamsvecs, output, plot=False)
