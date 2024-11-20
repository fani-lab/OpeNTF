import csv, requests, traceback, time
import pandas as pd
import os
import schedule #conda install --channel=conda-forge schedule

from team import Team


class Repo(Team):

    last_index = 0

    @staticmethod
    def check_previous_indexes(log_file):
        """

        Args:
            log_file: log file as csv

        Returns:
            int: index of the last crawled repo or 1 in case the log file does not exist
        """
        try:
            with open(log_file, mode='rb') as file:
                file.seek(-2, os.SEEK_END)
                while file.read(1) != b'\n':
                    file.seek(-2, os.SEEK_CUR)
                last_index = file.readline().decode().split(sep=',')[0]

            if 0 < int(last_index):
                return int(last_index)
            else:
                return 1

        except FileNotFoundError:
            print('Log file not found')
            with open(log_file, mode='w', newline='') as log_indexes:

                fieldnames = ['index', 'repo']
                writer = csv.DictWriter(log_indexes, fieldnames=fieldnames)
                writer.writeheader()
            return 1

    @staticmethod
    def crawl_repo(writer, reader, log_file, GET_header):
        repo = reader.readline().rstrip('\n')
        if not repo: return schedule.CancelJob
        try:
            print(f'Crawling {repo} ...')
            collabs = requests.get(f'https://api.github.com/repos/{repo}/contributors', headers=GET_header).json()
            langs = requests.get(f'https://api.github.com/repos/{repo}/languages', headers=GET_header).json()
            rels = requests.get(f'https://api.github.com/repos/{repo}/releases', headers=GET_header).json()

            general_repo_info = requests.get(f'https://api.github.com/repos/{repo}', headers=GET_header).json()
            try:
                stargazers_count = general_repo_info['stargazers_count']
                forks_count = general_repo_info['forks_count']
                created_at = general_repo_info['created_at']
                pushed_at = general_repo_info['pushed_at']
                if len(collabs) > 1:
                    writer.writerow( {'repo': repo, 'collabs': collabs, 'langs': langs, 'rels': rels, 'pushed_at': pushed_at,
                     'stargazers_count': stargazers_count, 'forks_count': forks_count, 'created_at': created_at})

                with open(log_file, 'a', newline='') as log_object:
                    # Writing to the log file
                    index_writer = csv.DictWriter(log_object, fieldnames=['index', 'repo'])
                    index_writer.writerow( {'index': Repo.last_index, 'repo': repo} )
                    Repo.last_index += 1

            except KeyError:
                print('The following error has occurred: {}'.format(general_repo_info['message']))

        except:
            print(f'Error occured for repo: {repo}')
            traceback.print_exc()

    @staticmethod
    def crawl_github(input, output,log_file, access_token, nseconds):
        """"
        data.csv is where the output of the data is stored

        repos.csv is the result of a repo list generated using BigQuery
        Steps to obtain this data:
        1. Ensure you have a Google account
        2. Open https://console.cloud.google.com/bigquery. Create a new project to enable access to the query.
        3. Open https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=github_repos to access the editor
        4. Type the following into the editor
        SELECT DISTINCT repo.name as repo FROM `githubarchive.day.2020*`
        WHERE type = 'MemberEvent'

        5. The program above will look at all github MemberEvent data from 2020 (starting on Jan 1). MemberEvents are recorded every time a new person becomes a contributor to a repo.
        6. Once result is generated, click save result and export to google drive and download the csv file as repos.csv.
        """
        # logging.basicConfig()
        # schedule_logger = logging.getLogger('schedule')
        # schedule_logger.setLevel(level=logging.DEBUG)

        header = {'Authorization': "Token " + access_token}
        Repo.last_index = Repo.check_previous_indexes(log_file)

        if Repo.last_index == 1:
            file_open_mode = 'w'
        else:
            file_open_mode = 'a'


        with open(input, 'r') as reader, open(output, file_open_mode, newline='') as output_csv:

            for i in range(1, Repo.last_index):
                reader.readline() #bypass the header 'repo' and crawled repos

            if file_open_mode == 'w':
                w = csv.DictWriter(output_csv, ['repo', 'collabs', 'langs', 'rels', 'stargazers_count', 'forks_count', 'created_at', 'pushed_at'])
                w.writeheader()
            else:
                w = csv.DictWriter(output_csv, ['repo', 'collabs', 'langs', 'rels', 'stargazers_count', 'forks_count', 'created_at', 'pushed_at'])

            job = schedule.every(nseconds).seconds.do(Repo.crawl_repo, writer=w, reader=reader, log_file=log_file, GET_header=header)  # NOT schedule.every(1).seconds.do(getData, w=w)
            while True:
                schedule.run_pending()
                if not schedule.jobs: return
                time.sleep(1)


Repo.crawl_github(input='./../../data/raw/gith/repos.csv', output='./../../data/raw/gith/data.csv', log_file='./../../data/raw/gith/log_index.csv', access_token='', nseconds=8)