import json
import traceback
import pickle
import time
import os
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
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
            with open(f'{output}/teams.pkl', 'rb') as infile:
                print("Loading teams pickle...")
                teams = pickle.load(infile)
            with open(f'{output}/indexes.pkl', 'rb') as infile:
                print("Loading indexes pickle...")
                i2m, m2i, i2s, s2i, i2t, t2i = pickle.load(infile)
            print(f"It took {time.time() - start_time} seconds to load from the pickles.")
            return i2m, m2i, i2s, s2i, i2t, t2i, teams
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
                                member.increase_n()
                            members.append(member)

                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team

                        # input_data.append(" ".join(team.get_skills()))
                        # output_data.append(" ".join([m.get_name() for m in team.members]))

                        counter += 1
                        if counter % 10000 == 0:
                            print(
                                f"{counter} instances have been loaded, and {time.time() - start_time} seconds has passed.")

                    except:  # ideally should happen only for the last line ']'
                        print(f'ERROR: There has been error in loading json line `{line}`!\n{traceback.format_exc()}')
                        continue
                        # raise
            if len(bypassed) > 0:
                with open(f'{output}/bypassed.json', 'w') as outfile:
                    json.dump(bypassed, outfile)
            print(f"It took {time.time() - start_time} seconds to load the data.")

            i2m, m2i = Team.build_index_members(all_members)
            i2s, s2i = Team.build_index_skills(teams)
            i2t, t2i = Team.build_index_teams(teams)
            write_time = time.time()
            with open(f'{output}/teams.pkl', "wb") as outfile:
                pickle.dump(teams, outfile)
            with open(f'{output}/indexes.pkl', "wb") as outfile:
                pickle.dump((i2m, m2i, i2s, s2i, i2t, t2i), outfile)
            with open(f'{output}/members.pkl', "wb") as outfile:
                pickle.dump(all_members, outfile)
            print(f"It took {time.time() - write_time} seconds to pickle the data")


            return i2m, m2i, i2s, s2i, i2t, t2i, teams

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



    @classmethod
    def get_stats(cls, teams, output):
        try:
            with open(f'{output}/stats_v2.pkl', 'rb') as infile:
                print("Loading the stats pickle ...")
                stats = pickle.load(infile)
                Publication.plot_stats(output)
                return stats

        except FileNotFoundError:
            print("File not found! Generating stats ...")

            stats = {
                'n_publications_per_n_skill': dict(),   # Number of publications per number of skills in a publication
                'n_publications_per_n_author': dict(),  # Number of publications per number of authors in a publication
                'n_publications_per_skill': dict(),  # Number of publications per skill
                'n_publications_per_author': dict(),  # Number of publications per author
                'n_publications_per_year': dict(),  # Number of publications per year
                'hist_n_skill_per_n_skill': Counter(),  # Histogram of number of skills per number of skills with the same number of publications
                'hist_n_author_per_n_author': Counter()  # Histogram of number of authors per number of authors with the same number of publications
            }

            start_time = time.time()
            # Iterating the teams (publications) to incrementally collect the stats
            for id, team in teams.items():
                skill_count = len(team.get_skills())
                author_count = len([m.get_id() for m in team.members])

                skill_pub_count = stats['n_publications_per_n_skill'].get(skill_count, 0)
                author_pub_count = stats['n_publications_per_n_author'].get(author_count, 0)

                stats['n_publications_per_n_skill'][skill_count] = skill_pub_count + 1
                stats['n_publications_per_n_author'][author_count] = author_pub_count + 1

                for skill in team.get_skills():
                    pub_count_of_skill = stats['n_publications_per_skill'].get(skill, 0)
                    stats['n_publications_per_skill'][skill] = pub_count_of_skill + 1
                for mem in team.members:
                    mem_key = f'{mem.get_id()}_{mem.get_name()}'
                    pub_count_of_author = stats['n_publications_per_author'].get(mem_key, 0)
                    stats['n_publications_per_author'][mem_key] = pub_count_of_author + 1

                year_count = stats['n_publications_per_year'].get(team.get_year(), 0)
                stats['n_publications_per_year'][team.get_year()] = year_count + 1

            skill_per_year = {k: set() for k in stats['n_publications_per_year'].keys()}  # Skills per year
            author_per_year = {k: set() for k in stats['n_publications_per_year'].keys()}  # Ids of authors per year

            n_skill_per_year = dict.fromkeys(stats['n_publications_per_year'].keys())  # Number of skills per year
            n_author_per_year = dict.fromkeys(stats['n_publications_per_year'].keys())  # Number of authors per year


            # Iterating the teams (publications) to incrementally collect the stats
            for id, team in teams.items():
                for skill in team.get_skills():
                    skill_per_year[team.get_year()].add(skill)
                for auth_id in [m.get_id() for m in team.members]:
                    author_per_year[team.get_year()].add(auth_id)

            # Number of publications per number of skills in a publication sorted ascendeingly by key
            stats['n_publications_per_n_skill'] = {k: v for k, v in sorted(stats['n_publications_per_n_skill'].items(), key=lambda item: item[0], reverse=False)}

            # Number of publications per number of authors in a publication sorted ascendeingly by key
            stats['n_publications_per_n_author'] = {k: v for k, v in sorted(stats['n_publications_per_n_author'].items(), key=lambda item: item[0], reverse=False)}

            # Number of publications per skill sorted ascendeingly by key
            stats['n_publications_per_skill'] = {k: v for k, v in sorted(stats['n_publications_per_skill'].items(), key=lambda item: item[0], reverse=False)}

            # Number of publications per author sorted ascendeingly by key
            stats['n_publications_per_author'] = {k: v for k, v in sorted(stats['n_publications_per_author'].items(), key=lambda item: item[0], reverse=False)}

            # Number of publications per year sorted ascendeingly by key
            stats['n_publications_per_year'] = {k: v for k, v in sorted(stats['n_publications_per_year'].items(), key=lambda item: item[0], reverse=False)}

            # Histogram of number of skills per number of skills with the same number of publications sorted ascendeingly by key
            stats['hist_n_skill_per_n_skill'] = {k: v for k, v in sorted(dict(Counter(list(stats['n_publications_per_skill'].values()))).items(), key=lambda item: item[0], reverse=False)}

            # Histogram of number of authors per number of authors with the same number of publications sorted ascendeingly by key
            stats['hist_n_author_per_n_author'] = {k: v for k, v in sorted(dict(Counter(list(stats['n_publications_per_author'].values()))).items(), key=lambda item: item[0], reverse=False)}

            # Number of skills per year sorted ascendeingly by key
            n_skill_per_year = {k: len(v) for k, v in skill_per_year.items()}
            stats['n_skill_per_year'] = {k: v for k, v in
                                         sorted(n_skill_per_year.items(), key=lambda item: item[0], reverse=False)}
            # Number of authors per year sorted ascendeingly by key
            n_author_per_year = {k: len(v) for k, v in author_per_year.items()}
            stats['n_author_per_year'] = {k: v for k, v in sorted(n_author_per_year.items(), key=lambda item: item[0], reverse=False)}

            print(f"It took {time.time() - start_time} seconds to get the stats.")
            write_time = time.time()
            with open(f'{output}/stats_v2.json', 'w') as outfile:
                json.dump(stats, outfile)
            with open(f'{output}/stats_v2.pkl', 'wb') as outfile:
                pickle.dump(stats, outfile)

            print(f"It took {time.time() - write_time} seconds to pickle the stats")
            print(f"It took {time.time() - start_time} seconds to get the stats and pickle it.")
        Publication.plot_stats(output)
        return stats

    @staticmethod
    def plot_stats(output):
        try:
            with open(f'{output}/stats.pkl', 'rb') as infile:
                print("Loading the stat pickle...")
                stats = pickle.load(infile)

            fig1 = plt.figure()
            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.bar(*zip(*stats['n_publications_per_n_skill'].items()))
            ax1.set_xlabel('Number of skills')
            ax1.set_ylabel('Number of papers')
            fig1.savefig(f'{output}/n_publications_per_n_skill.png', dpi=100, bbox_inches='tight')

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 1, 1)
            ax2.bar(*zip(*stats['n_publications_per_n_author'].items()))
            ax2.set_xlabel('Number of authors')
            ax2.set_ylabel('Number of papers')
            fig2.savefig(f'{output}/n_publications_per_n_author.png', dpi=100, bbox_inches='tight')

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1, 1, 1)
            ax3.bar(*zip(*stats['n_publications_per_year'].items()))
            ax3.set_xlabel('Years')
            ax3.set_ylabel('Number of papers')
            fig3.savefig(f'{output}/n_publications_per_year.png', dpi=100, bbox_inches='tight')

            fig4 = plt.figure()
            ax4 = fig4.add_subplot(1, 1, 1)
            ax4.bar(*zip(*stats['hist_n_skill_per_n_skill'].items()))
            ax4.set_xlabel('Number of publications of skills with the same number of publications')
            ax4.set_ylabel('Number of skills')
            fig4.savefig(f'{output}/hist_n_skill_per_n_skill.png', dpi=100, bbox_inches='tight')

            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1, 1, 1)
            ax5.bar(*zip(*stats['hist_n_author_per_n_author'].items()))
            ax5.set_xlabel('Number of publications of authors with the same number of publications')
            ax5.set_ylabel('Number of authors')
            fig5.savefig(f'{output}/hist_n_author_per_n_author.png', dpi=100, bbox_inches='tight')

            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1, 1, 1)
            ax6.bar(*zip(*stats['n_skill_per_year'].items()))
            ax6.set_xlabel('Years')
            ax6.set_ylabel('Number of skills')
            fig6.savefig(f'{output}/n_skill_per_year.png', dpi=100, bbox_inches='tight')

            fig7 = plt.figure()
            ax7 = fig7.add_subplot(1, 1, 1)
            ax7.bar(*zip(*stats['n_author_per_year'].items()))
            ax7.set_xlabel('Years')
            ax7.set_ylabel('Number of authors')
            fig7.savefig(f'{output}/n_author_per_year.png', dpi=100, bbox_inches='tight')

            plt.show()

        except FileNotFoundError:
            print("File not found!")

    @staticmethod
    def get_unigram(output, m2i):
        try:
            with open(f'{output}/stats.pkl', 'rb') as infile:
                print("Loading the stat pickle...")
                stats = pickle.load(infile)

            n_papers = sum(list(stats['n_publications_per_year'].values()))
            n_authors = len(list(stats['n_publications_per_author'].values()))

            unigram = np.zeros(n_authors)
            for k, v in stats['n_publications_per_author'].items():
                unigram[m2i[k]] = v/n_papers

            return unigram


        except FileNotFoundError:
            print("File not found!")