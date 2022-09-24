import json, os
from tqdm import tqdm
from time import time

from cmn.author import Author
from cmn.team import Team

class Publication(Team):
    def __init__(self, id, authors, title, datetime, doc_type, venue, references, fos, keywords):
        super().__init__(id, authors, None, datetime, venue)
        self.title = title
        self.doc_type = doc_type
        self.location = venue #e.g., {'raw': 'international conference on human-computer interaction', 'id': 1127419992, 'type': 'c'}
        self.references = references
        self.fos = fos #field of study
        self.keywords = keywords
        self.skills = self.set_skills()

        for author in self.members:
            author.teams.add(self.id)
            author.skills.update(set(self.skills))
        self.members_locations = [(venue['id'], venue['id'], venue['id'])] * len(self.members)


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
            return super(Publication, Publication).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} (progress in bytes) ...")
            teams = {}; candidates = {}

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
                        venue = jsonline['venue'] if 'venue' in jsonline.keys() else None
                        references = jsonline['references'] if 'references' in jsonline.keys() else []
                        keywords = jsonline['keywords'] if 'keywords' in jsonline.keys() else []

                        # a team must have skills and members
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
                                candidates[idname] = Author(member_id, member_name, member_org)
                            members.append(candidates[idname])
                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team
                        if 'nrow' in settings['domain']['dblp'].keys() and len(teams) > settings['domain']['dblp']['nrow']: break
                    except json.JSONDecodeError as e:  # ideally should happen only for the last line ']'
                        print(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                        continue
                    except Exception as e:
                        raise e
            return super(Publication, Publication).read_data(teams, output, filter, settings)
        except Exception as e:
            raise e

    # @classmethod
    # def get_stats(cls, teamsvecs, output, plot=False):
        # return super(Publication, cls).get_stats(teamsvecs, output, plot=False)
