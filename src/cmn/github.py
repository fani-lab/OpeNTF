import pandas as pd
from tqdm import tqdm

from src.cmn.team import Team
from src.cmn.developer import Developer


class Repo(Team):

    def __init__(self, idx: int, contributors: list, repo_name: str, languages_and_lines: list, forks_count: int,
                 stargazers_count: int, created_at: str, pushed_at: str, ncontributions: list, releases: list):

        super().__init__(id=idx, members=contributors, skills=languages_and_lines, datetime=created_at)
        self.repo_name = repo_name
        self.forks_count = forks_count
        self.stargazers_count = stargazers_count
        self.pushed_at = pushed_at
        self.ncontributions = ncontributions
        self.releases = releases

    @staticmethod
    def read_data(datapath, output, index, filter, settings):


        try:
            return super(Repo, Repo).load_data(output, index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} ...")

            raw_dataset = pd.read_csv(datapath, converters={'collabs': eval, 'langs': eval, 'rels': eval})
            dict_of_teams = dict()

            try:
                for idx, row in tqdm(raw_dataset.iterrows(), total=len(raw_dataset)):
                    contributors = row['collabs']
                    list_of_developers = list()
                    list_of_contributions = list()

                    for contributor in contributors:
                        developer_object = Developer(name=contributor['login'], id=contributor['id'], url=contributor['url'])
                        list_of_developers.append(developer_object)
                        list_of_contributions.append(contributor['contributions'])

                    repo_name = row['repo']
                    languages_and_lines = list(row['langs'].items())
                    stargazers_count = row['stargazers_count']
                    forks_count = row['forks_count']
                    created_at = row['created_at']
                    pushed_at = row['pushed_at']
                    ncontributions = row['collabs']
                    releases = row['rels']

                    dict_of_teams[idx] = Repo(idx=idx, contributors=list_of_developers, repo_name=repo_name, releases=releases,
                                              languages_and_lines=languages_and_lines, stargazers_count=stargazers_count,
                                              forks_count=forks_count, created_at=created_at, pushed_at=pushed_at, ncontributions=ncontributions)
            except Exception as e:
                raise e

            return super(Repo, Repo).read_data(dict_of_teams, output, filter, settings)
