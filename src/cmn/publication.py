import json
import traceback
import pickle
import time
import os
from collections import Counter

from cmn.member import Member
from cmn.author import Author
from cmn.team import Team

class Publication(Team):
    def __init__(self, id, authors, title, year, doc_type, venue, references, fos, keywords):
        super().__init__(id, authors)
        self.title = title
        self.year = year
        self.doc_type = doc_type
        self.venue = venue
        self.references = references
        self.fos = fos
        self.keywords = keywords
        self.skills = self.set_skills()

    # Fill the fields attribute with non-zero weight from FOS
    def set_skills(self):
        skills = []
        for skill in self.fos:
            if skill["w"] != 0.0:
                skills.append(skill["name"].replace(" ", "_"))
        # Extend the fields with keywords
        if len(self.keywords) != 0:
            skills.extend(self.keywords.replace(" ", "_"))
        return skills

    def get_skills(self):
        return self.skills

    def get_year(self):
        return self.year

    @staticmethod
    def read_data(data_path, output, topn=None):
        counter = 0
        teams = {}
        all_members = {}
        bypassed = {}
        try:
            start_time = time.time()
            with open(f'{output}teams.pkl', 'rb') as infile:
                print("Loading teams pickle...")
                teams = pickle.load(infile)
            with open(f'{output}indexes.pkl', 'rb') as infile:
                print("Loading indexes pickle...")
                i2m, m2i, i2s, s2i = pickle.load(infile)
            print(f"It took {time.time() - start_time} seconds to load from the pickles.")
            return i2m, m2i, i2s, s2i, teams
        except:
            print("Pickles not found! Reading raw data ...")
            with open(data_path, "r", encoding='utf-8') as jf:
                # Skip the first line
                jf.readline()
                while True:
                    try:
                        # Read line by line to not overload the memory
                        line = jf.readline()
                        if not line or (topn and counter >= topn):
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
                        if 'fos' in jsonline.keys():
                            fos = jsonline['fos']
                        else:
                            bypassed[id] = "No FOS!"
                            continue  # fos -> skills
                        if 'authors' not in jsonline.keys():
                            bypassed[id] = "No Author!"
                            continue

                        members = []
                        for auth in jsonline['authors']:

                            # Retrieve the desired attributes
                            member_id = auth['id']
                            member_name = auth['name'].replace(" ", "_")

                            if 'org' in auth.keys():
                                member_org = auth['org'].replace(" ", "_")
                            else:
                                member_org = ""

                            if member_id not in all_members.keys():
                                member = Author(member_id, member_name, member_org)
                                all_members[member_id] = member
                            else:
                                member = all_members[member_id]
                            members.append(member)

                            # member = Author(member_id, member_name, member_org)
                            # members.append(member)
                            # if member_id not in members.keys():
                            #     members[member_id] = member

                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team

                        # input_data.append(" ".join(team.get_skills()))
                        # output_data.append(" ".join(team.get_members_names()))

                        counter += 1
                        if counter % 10000 == 0:
                            print(
                                f"{counter} instances have been loaded, and {time.time() - start_time} seconds has passed.")

                    except:  # ideally should happen only for the last line ']'
                        print(f'ERROR: There has been error in loading json line `{line}`!\n{traceback.format_exc()}')
                        continue
                        # raise
            output_name = "_".join(data_path.split("/")[-1].split(".")[:-1])
            if len(bypassed) > 0:
                with open(f'../data/preprocessed/{output_name}/bypassed.json', 'w') as outfile:
                    json.dump(bypassed, outfile)
            print(f"It took {time.time() - start_time} seconds to load the data.")

            i2m, m2i = Team.build_index_members(all_members)
            i2s, s2i = Team.build_index_skills(teams)

            write_time = time.time()
            with open(f'{output}teams.pkl', "wb") as outfile:
                pickle.dump(teams, outfile)
            with open(f'{output}indexes.pkl', "wb") as outfile:
                pickle.dump((i2m, m2i, i2s, s2i), outfile)
            print(f"It took {time.time() - write_time} seconds to pickle the data")

            return i2m, m2i, i2s, s2i, teams

    @classmethod
    def get_stats(cls, teams, output):
        try:
            with open(output, 'rb') as infile:
                print("Loading the stat pickle...")
                return pickle.load(infile)

        except FileNotFoundError:
            print("File not found! Generating stats ...")

            #please change the stat names based on what I explained in msteams.
            #also, we need to split these stats. some must go to team.py some stays here like those related to yearly. We'll talk.
            #also where are the codes that plot them?

            stats = {
                'skill_count_of_pub_count': dict(),# Count of publications with different count of skills
                'author_count_of_pub_count': dict(),# Count of publications with different count of authors
                'pub_count_of_skills': dict(),  # Count of publications for each skill
                'pub_count_of_authors': dict(),  # Count of publications for each author
                'pub_count_of_years': dict(),  # Count of publications for each year
                'skill_count_of_count_of_pub': Counter(),  # Count of skills for different count of publications
                'author_count_of_count_of_pub': Counter()  # Count of authors for different count of publications
            }

            start_time = time.time()
            # Iterating the teams (publications) to incrementally collect the stats
            for id, team in teams.items():
                skill_count = len(team.get_skills())
                author_count = len(team.get_members_names())

                skill_pub_count = stats['skill_count_of_pub_count'].get(skill_count, 0)
                author_pub_count = stats['author_count_of_pub_count'].get(author_count, 0)

                stats['skill_count_of_pub_count'][skill_count] = skill_pub_count + 1
                stats['author_count_of_pub_count'][author_count] = author_pub_count + 1

                for skill in team.get_skills():
                    pub_count_of_skill = stats['pub_count_of_skills'].get(skill, 0)
                    stats['pub_count_of_skills'][skill] = pub_count_of_skill + 1
                for mem in team.get_members_names():
                    pub_count_of_author = stats['pub_count_of_authors'].get(mem, 0)
                    stats['pub_count_of_authors'][mem] = pub_count_of_author + 1

                year_count = stats['pub_count_of_years'].get(team.get_year(), 0)
                stats['pub_count_of_years'][team.get_year()] = year_count + 1

            # Count of publications with different count of skills sorted descendingly
            stats['skill_count_of_pub_count'] = {k: v for k, v in sorted(stats['skill_count_of_pub_count'].items(), key=lambda item: item[1], reverse=True)}

            # Count of publications with different count of authors sorted descendingly
            stats['author_count_of_pub_count'] = {k: v for k, v in sorted(stats['author_count_of_pub_count'].items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each skill sorted descendingly
            stats['pub_count_of_skills'] = {k: v for k, v in sorted(stats['pub_count_of_skills'].items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each author sorted descendingly
            stats['pub_count_of_authors'] = {k: v for k, v in sorted(stats['pub_count_of_authors'].items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each year sorted descendingly
            stats['pub_count_of_years'] = {k: v for k, v in sorted(stats['pub_count_of_years'].items(), key=lambda item: item[1], reverse=True)}

            # Count of skills for different count of publications sorted descendingly
            stats['skill_count_of_count_of_pub'] = {k: v for k, v in sorted(dict(Counter(list(stats['pub_count_of_skills'].values()))).items(), key=lambda item: item[1], reverse=True)}

            # Count of authors for different count of publications sorted descendingly
            stats['author_count_of_count_of_pub'] = {k: v for k, v in sorted(dict(Counter(list(stats['pub_count_of_authors'].values()))).items(), key=lambda item: item[1], reverse=True)}

            print(f"It took {time.time() - start_time} seconds to get the stats.")
            write_time = time.time()
            with open(output, 'w') as outfile:
                json.dump(stats, outfile)

            print(f"It took {time.time() - write_time} seconds to pickle the stats")
            print(f"It took {time.time() - start_time} seconds to get the stats and pickle it.")
        return stats
