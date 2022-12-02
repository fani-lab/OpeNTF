import pandas as pd
from tqdm import tqdm
from cmn.team import Team
from cmn.developer import Developer


class Repo(Team):

    def __init__(self, idx: int, contributors: list, name: str, languages_lines: list, nforks: int,
                 nstargazers: int, created_at: str, year: int, pushed_at: str, ncontributions: list, releases: list):

        super().__init__(id=idx, members=contributors, skills=None, datetime=year)
        self.name = name
        self.nforks = nforks
        self.nstargazers = nstargazers
        self.created_at = created_at
        self.pushed_at = pushed_at
        self.ncontributions = ncontributions
        self.releases = releases
        self.languages_lines = languages_lines
        self.skills = self.set_skills()
        for dev in self.members:
            dev.teams.add(self.id)
            dev.skills.update(set(self.skills))

    def set_skills(self):
        skills = set()
        top_langs = {'python', 'java', 'c++', 'go', 'javascript', 'typescript', 'php', 'ruby', 'c', 'c#', 'nix', 'scala', 'shell', 'kotlin', 'rust', 'dart', 'swift', 'dm', 'systemverilog', 'lua'}
        for language in self.languages_lines:
            if language[0].lower() in top_langs:
                skills.add(language[0].lower()) 
        return skills

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        try:
            return super(Repo, Repo).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} ...")
            raw_dataset = pd.read_csv(datapath, converters={'collabs': eval, 'langs': eval, 'rels': eval}, encoding='latin-1')
            dict_of_teams = dict(); repos = dict(); candidates = dict()
            raw_dataset['created_at'] = pd.to_datetime(raw_dataset['created_at'])
            raw_dataset['year'] = raw_dataset['created_at'].dt.year
            try:
                for idx, row in tqdm(raw_dataset.iterrows(), total=len(raw_dataset)):
                    contributors = row['collabs']
                    list_of_developers = list()
                    list_of_contributions = list()

                    for contributor in contributors:
                        if isinstance(contributor, str): continue
                        if (idname := f"{contributor['id']}_{contributor['login']}") not in candidates:
                            candidates[idname] = Developer(name=contributor['login'], id=contributor['id'], url=contributor['url'])
                        list_of_developers.append(candidates[idname])
                        list_of_contributions.append(contributor['contributions'])

                    repo_name = row['repo']
                    languages_lines = list(row['langs'].items())
                    nstargazers = row['stargazers_count']
                    nforks = row['forks_count']
                    created_at = row['created_at']
                    year = row['year']
                    pushed_at = row['pushed_at']
                    ncontributions = row['collabs']
                    releases = row['rels']

                    if repo_name not in repos:
                        dict_of_teams[idx] = Repo(idx=idx, contributors=list_of_developers, name=repo_name, releases=releases,
                                                  languages_lines=languages_lines, nstargazers=nstargazers,
                                                  nforks=nforks, created_at=created_at, year=year, pushed_at=pushed_at, ncontributions=ncontributions)
                        repos[repo_name] = dict_of_teams[idx]
                    else:
                        pass

            except Exception as e: raise e

            return super(Repo, Repo).read_data(dict_of_teams, output, filter, settings)
