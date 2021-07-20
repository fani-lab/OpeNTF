import json
import traceback
import random
from cmn.member import Member
from cmn.author import Author
from cmn.team import Team
import pickle
import time
import os
from collections import Counter

random.seed(0)


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
    def read_data(data_path, topn=None):
        counter = 0
        teams = {}
        all_members = {}
        # input_data = []
        # output_data = []
        output_name = "_".join(data_path.split("/")[-1].split(".")[:-1])
        if not os.path.isdir(f'../data/preprocessed/{output_name}/'): os.mkdir(f'../data/preprocessed/{output_name}/')
        output_pickle = f'../data/preprocessed/{output_name}/teams.pickle'

        start_time = time.time()
        try:
            with open(output_pickle, 'rb') as infile:
                print("Loading the pickle.")
                all_members, teams = pickle.load(infile)
                print(f"It took {time.time() - start_time} seconds to load from the pickle.")
                return all_members, teams
        except:
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
                            continue  # fos -> skills
                        if 'authors' not in jsonline.keys(): continue

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
            print(f"It took {time.time() - start_time} seconds to load the data.")
            write_time = time.time()
            with open(output_pickle, "wb") as outfile:
                pickle.dump((all_members, teams), outfile)
            print(f"It took {time.time() - write_time} seconds to pickle the data")
            print(f"It took {time.time() - start_time} seconds to load the data and pickle it.")
        return all_members, teams

    @staticmethod
    def get_stats(teams, data_path):
        output_name = "_".join(data_path.split("/")[-1].split(".")[:-1])
        output_pickle = f'../data/preprocessed/{output_name}/stats.pickle'
        output_path = f'../data/preprocessed/{output_name}/'
        start_time = time.time()
        try:
            with open(output_pickle, 'rb') as infile:
                print("Loading the pickle.")
                skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub = pickle.load(infile)
                print(f"It took {time.time() - start_time} seconds to load from the pickle.")
                return skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub

        except FileNotFoundError:
            skill_count_of_pub_count = dict()  # Count of publications with different count of skills
            author_count_of_pub_count = dict()  # Count of publications with different count of authors
            pub_count_of_skills = dict()  # Count of publications for each skill
            pub_count_of_authors = dict()  # Count of publications for each author
            pub_count_of_years = dict()  # Count of publications for each year
            skill_count_of_count_of_pub = Counter()  # Count of skills for different count of publications
            author_count_of_count_of_pub = Counter()  # Count of authors for different count of publications

            # Iterating the teams (publications) to incrementally collect the stats
            for id, team in teams.items():
                skill_count = len(team.get_skills())
                author_count = len(team.get_members_names())

                skill_pub_count = skill_count_of_pub_count.get(skill_count, 0)
                author_pub_count = author_count_of_pub_count.get(author_count, 0)

                skill_count_of_pub_count[skill_count] = skill_pub_count + 1
                author_count_of_pub_count[author_count] = author_pub_count + 1

                for skill in team.get_skills():
                    pub_count_of_skill = pub_count_of_skills.get(skill, 0)
                    pub_count_of_skills[skill] = pub_count_of_skill + 1
                for mem in team.get_members_names():
                    pub_count_of_author = pub_count_of_authors.get(mem, 0)
                    pub_count_of_authors[mem] = pub_count_of_author + 1

                year_count = pub_count_of_years.get(team.get_year(), 0)
                pub_count_of_years[team.get_year()] = year_count + 1

            # Count of publications with different count of skills sorted descendingly
            skill_count_of_pub_count = {k: v for k, v in sorted(skill_count_of_pub_count.items(), key=lambda item: item[1], reverse=True)}

            # Count of publications with different count of authors sorted descendingly
            author_count_of_pub_count = {k: v for k, v in sorted(author_count_of_pub_count.items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each skill sorted descendingly
            pub_count_of_skills = {k: v for k, v in sorted(pub_count_of_skills.items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each author sorted descendingly
            pub_count_of_authors = {k: v for k, v in sorted(pub_count_of_authors.items(), key=lambda item: item[1], reverse=True)}

            # Count of publications for each year sorted descendingly
            pub_count_of_years = {k: v for k, v in sorted(pub_count_of_years.items(), key=lambda item: item[1], reverse=True)}

            # Count of skills for different count of publications sorted descendingly
            skill_count_of_count_of_pub = {k: v for k, v in sorted(dict(Counter(list(pub_count_of_skills.values()))).items(), key=lambda item: item[1], reverse=True)}

            # Count of authors for different count of publications sorted descendingly
            author_count_of_count_of_pub = {k: v for k, v in sorted(dict(Counter(list(pub_count_of_authors.values()))).items(), key=lambda item: item[1], reverse=True)}

            s_c_o_p_c_path = f'{output_path}/s_c_o_p_c.json'
            if not os.path.isfile(s_c_o_p_c_path):
                with open(s_c_o_p_c_path, 'w') as outfile:
                    json.dump(skill_count_of_pub_count, outfile)

            a_c_o_p_c_path = f'{output_path}/a_c_o_p_c.json'
            if not os.path.isfile(a_c_o_p_c_path):
                with open(a_c_o_p_c_path, 'w') as outfile:
                    json.dump(author_count_of_pub_count, outfile)

            p_c_o_s_path = f'{output_path}/p_c_o_s.json'
            if not os.path.isfile(p_c_o_s_path):
                with open(p_c_o_s_path, 'w') as outfile:
                    json.dump(pub_count_of_skills, outfile)

            p_c_o_a_path = f'{output_path}/p_c_o_a.json'
            if not os.path.isfile(p_c_o_a_path):
                with open(p_c_o_a_path, 'w') as outfile:
                    json.dump(pub_count_of_authors, outfile)

            p_c_o_y_path = f'{output_path}/p_c_o_y.json'
            if not os.path.isfile(p_c_o_y_path):
                with open(p_c_o_y_path, 'w') as outfile:
                    json.dump(pub_count_of_years, outfile)

            s_c_o_c_o_p_path = f'{output_path}/s_c_o_c_o_p.json'
            if not os.path.isfile(s_c_o_c_o_p_path):
                with open(s_c_o_c_o_p_path, 'w') as outfile:
                    json.dump(skill_count_of_count_of_pub, outfile)

            a_c_o_c_o_p_path = f'{output_path}/a_c_o_c_o_p.json'
            if not os.path.isfile(a_c_o_c_o_p_path):
                with open(a_c_o_c_o_p_path, 'w') as outfile:
                    json.dump(author_count_of_count_of_pub, outfile)

            print(f"It took {time.time() - start_time} seconds to get the stats.")
            write_time = time.time()
            with open(output_pickle, "wb") as outfile:
                pickle.dump((skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub), outfile)
            print(f"It took {time.time() - write_time} seconds to pickle the stats")
            print(f"It took {time.time() - start_time} seconds to get the stats and pickle it.")
        return skill_count_of_pub_count, author_count_of_pub_count, pub_count_of_skills, pub_count_of_authors, pub_count_of_years, skill_count_of_count_of_pub, author_count_of_count_of_pub
