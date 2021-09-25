import json
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
            author.skills.union(set(self.skills))

    # Fill the fields attribute with non-zero weight from FOS
    def set_skills(self):
        skills = set()
        for skill in self.fos:
            if skill["w"] != 0.0:
                skills.add(skill["name"].replace(" ", "_"))
        # Extend the fields with keywords
        if len(self.keywords):
            skills.union(set([keyword.replace(" ", "_") for keyword in self.keywords]))
        return skills

    def get_skills(self):
        return self.skills

    def get_year(self):
        return self.year

    @staticmethod
    def read_data(datapath, output, index=True, topn=None):
        try:
            st = time()
            print("Loading indexes pickle...")
            with open(f'{output}/indexes.pkl', 'rb') as infile: i2c, c2i, i2s, s2i, i2t, t2i = pickle.load(infile)
            print(f"It took {time() - st} seconds to load from the pickles.")
            teams = None
            if not index:
                st = time()
                print("Loading teams pickle...")
                with open(f'{output}/teams.pkl', 'rb') as tfile: teams = pickle.load(tfile)
                print(f"It took {time() - st} seconds to load from the pickles.")

            return i2c, c2i, i2s, s2i, i2t, t2i, teams
        except (FileNotFoundError, EOFError) as e:
            print("Pickles not found! Reading raw data ...")
            teams = {}; candidates = {}
            n_row = 0
            with open(datapath, "r", encoding='utf-8') as jf:
                # Skip the first line
                jf.readline()
                while True:
                    try:
                        # Read line by line to not overload the memory
                        line = jf.readline()
                        n_row += 1
                        if not line or (topn and n_row >= topn):
                            break

                        jsonline = json.loads(line.lower().lstrip(","))
                        # Retrieve the desired attributes
                        id = jsonline['id']
                        title = jsonline['title']
                        year = jsonline['year']
                        type = jsonline['doc_type']
                        venue = jsonline['venue'] if 'venue' in jsonline.keys() else None
                        references = jsonline['references'] if 'references' in jsonline.keys() else []
                        keywords = jsonline['keywords'] if 'keywords' in jsonline.keys() else []

                        # a team must have skills and members
                        try: fos = jsonline['fos']
                        except: print(f'Warning! No fos for team id={id}. Bypassed!'); continue  #publication must have fos (skills)
                        try: authors = jsonline['authors']
                        except: print(f'Warning! No author for team id={id}. Bypassed!'); continue #publication must have authors (members)

                        members = []
                        for author in authors:
                            member_id = author['id']
                            member_name = author['name'].replace(" ", "_")
                            member_org = author['org'].replace(" ", "_") if 'org' in author else ""
                            if (idname := f'{member_id}_{member_name}') not in candidates:
                                candidates[idname] = Author(member_id, member_name, member_org)
                            members.append(candidates[idname])

                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team
                        if n_row % 10000 == 0: print(f"{n_row} instances have been loaded, and {time() - st} seconds has passed.")

                    except json.JSONDecodeError as e:  # ideally should happen only for the last line ']'
                        print(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                        continue
                    except Exception as e:
                        raise e

            print(f"It took {time() - st} seconds to load the data. #teams: {len(teams)} out of #lines: {n_row}.")

            i2c, c2i = Team.build_index_candidates(teams.values())
            i2s, s2i = Team.build_index_skills(teams.values())
            i2t, t2i = Team.build_index_teams(teams.values())
            st = time()
            try: os.makedirs(output)
            except FileExistsError as ex: pass
            with open(f'{output}/teams.pkl', "wb") as outfile: pickle.dump(teams, outfile)
            with open(f'{output}/indexes.pkl', "wb") as outfile: pickle.dump((i2c, c2i, i2s, s2i, i2t, t2i), outfile)
            with open(f'{output}/candidates.pkl', "wb") as outfile: pickle.dump(candidates, outfile)
            print(f"It took {time() - st} seconds to pickle the data")

            return i2c, c2i, i2s, s2i, i2t, t2i, teams
        except Exception as e:
            raise e

    # @classmethod
    # def get_stats(cls, teamsvecs, output, plot=False):
        # return super(Publication, cls).get_stats(teamsvecs, output, plot=False)

    @staticmethod
    def remove_outliers(output, n):
        with open(f'{output}/members.pkl', 'rb') as infile:
            all_members = dict(pickle.load(infile))
            for id in [member.get_id() for member in all_members.values() if member.get_n_papers() <= n]: del all_members[id]
            # for member in all_members.values():
            #     if member.get_n_papers() <= n:
            #         del all_members[member.get_id()]
        with open(f'{output}/teams.pkl', 'rb') as infile:
            teams = dict(pickle.load(infile))
            for team in teams.values():
                new_team_members = [member for member in team.members if member.get_id() in all_members.keys()]
                team.set_members(new_team_members)

        i2m, m2i = Team.build_index_members(all_members)
        i2s, s2i = Team.build_index_skills(teams)
        i2t, t2i = Team.build_index_teams(teams)

        write_time = time.time()
        with open(f'{output}/teams_v2.pkl', "wb") as outfile:
            pickle.dump(teams, outfile)
        with open(f'{output}/indexes_v2.pkl', "wb") as outfile:
            pickle.dump((i2m, m2i, i2s, s2i, i2t, t2i), outfile)
        with open(f'{output}/members_v2.pkl', "wb") as outfile:
            pickle.dump(all_members, outfile)
        print(f"It took {time.time() - write_time} seconds to pickle the data")
        return i2m, m2i, i2s, s2i, i2t, t2i, teams
