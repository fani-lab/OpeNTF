import csv, requests, traceback, time
import schedule #conda install --channel=conda-forge schedule


from team import Team


class Repo(Team):

    @staticmethod
    def crawl_repo(writer, reader, GET_header):
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
                if len(collabs) > 1: writer.writerow(
                    {'repo': repo, 'collabs': collabs, 'langs': langs, 'rels': rels, 'pushed_at': pushed_at,
                     'stargazers_count': stargazers_count, 'forks_count': forks_count, 'created_at': created_at})
            except KeyError:
                print('The following error has occurred: {}'.format(general_repo_info['message']))

        except:
            print(f'Error occured for repo: {repo}')
            traceback.print_exc()

    @staticmethod
    def crawl_github(input, output, access_token, nseconds):
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
        with open(input, 'r') as reader, open(output, 'w', newline='') as output_csv:
            reader.readline() #bypass the header 'repo'
            w = csv.DictWriter(output_csv, ['repo', 'collabs', 'langs', 'rels', 'stargazers_count', 'forks_count', 'created_at', 'pushed_at'])
            w.writeheader()
            job = schedule.every(nseconds).seconds.do(Repo.crawl_repo, writer=w, reader=reader, GET_header=header)  # NOT schedule.every(1).seconds.do(getData, w=w)
            while True:
                schedule.run_pending()
                if not schedule.jobs: return
                time.sleep(1)


Repo.crawl_github(input='./../../data/raw/gith/toy.repos.csv', output='./../../data/raw/gith/toy.data.csv', access_token='', nseconds=8)