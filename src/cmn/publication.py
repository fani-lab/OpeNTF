import json, os, logging
log = logging.getLogger(__name__)

import pkgmgr as opentf
from .author import Author
from .team import Team

class Publication(Team):
    def __init__(self, id, authors, title, datetime, doc_type, venue, references, fos, keywords):
        super().__init__(id=id, members=authors, skills=set(), datetime=datetime, location=venue)
        self.title = title
        self.doc_type = doc_type
        self.location = venue #e.g., {'raw': 'international conference on human-computer interaction', 'id': 1127419992, 'type': 'c'}
        self.references = references
        self.fos = fos #field of study, e.g., [{"name":"Deep Learning","w":0.41115},{"name":"Image Segmentation","w":0.49814}]
        self.keywords = keywords

        self.skills = {f['name'].replace(' ', '_').lower() for f in self.fos}  # TODO: ordered skills based on skill["w"]
        # Extend the skills with keywords
        # if len(self.keywords): skills.union(set([keyword.replace(" ", "_") for keyword in self.keywords]))

        for author in self.members:
            author.teams.add(self.id)
            author.skills.update(set(self.skills))
        self.members_locations = [(str(venue), str(venue), str(venue))] * len(self.members) # cfg.location should be set to 'country' or 'venue'


    # Fill the fields attribute with non-zero weight from FOS
    def set_skills(self):
        skills = set()

        return skills

    @staticmethod
    def read_data(datapath, output, cfg, indexes_only=False):
        tqdm = opentf.install_import('tqdm==4.65.0', 'tqdm', 'tqdm')
        try: return super(Publication, Publication).load_data(output, indexes_only)
        except (FileNotFoundError, EOFError) as e:
            log.info(f'Pickles not found! Reading raw data from {datapath} (progress in bytes) ...')
            teams = {}; candidates = {}

            with tqdm(total=os.path.getsize(datapath)) as pbar, open(datapath, 'r', encoding='utf-8') as jf:
                for line in jf:
                    try:
                        if not line: break
                        # Skip lines that are just brackets (for JSON array format files)
                        if line.strip() in ['[', ']']: pbar.update(len(line)); continue
                        pbar.update(len(line))
                        jsonline = json.loads(line.lower().lstrip(","))
                        id = jsonline['id']
                        title = jsonline['title'].lower()
                        year = jsonline['year']
                        type = jsonline['doc_type'].lower()
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
                            member_name = author['name'].replace(' ', '_').lower()
                            member_org = author['org'].replace(' ', '_').lower() if 'org' in author else ''
                            if (idname := f'{member_id}_{member_name}') not in candidates: candidates[idname] = Author(member_id, member_name, member_org)
                            members.append(candidates[idname])
                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team
                        if 'nrow' in cfg and len(teams) > cfg.nrow: break
                    except json.JSONDecodeError as e:  # ideally should happen only for the last line ']'
                        log.error(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                        continue
                    except Exception as e: raise e
            return super(Publication, Publication).read_data(teams, output, cfg)
        except Exception as e: raise e