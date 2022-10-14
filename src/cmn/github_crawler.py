import csv, requests, traceback, schedule, time


def get_data(w):
    global repo
    repo = repo.rstrip('\n')
    try:

        collabs = requests.get(f'https://api.github.com/repos/{repo}/contributors', headers=headers).json()
        langs = requests.get(f'https://api.github.com/repos/{repo}/languages', headers=headers).json()
        rels = requests.get(f'https://api.github.com/repos/{repo}/releases', headers=headers).json()

        general_repo_info = requests.get(f'https://api.github.com/repos/{repo}', headers=headers).json()
        stargazers_count = general_repo_info['stargazers_count']
        forks_count = general_repo_info['forks_count']
        created_at = general_repo_info['created_at']
        pushed_at = general_repo_info['pushed_at']

        if len(collabs) > 1:
            w.writerow({'repo': repo, 'collabs': collabs, 'langs': langs, 'rels': rels, 'pushed_at': pushed_at,
                        'stargazers_count': stargazers_count, 'forks_count': forks_count, 'created_at': created_at})
    except:
        print(f'Error occured for repo: {repo}')
        traceback.print_exc()
    repo = reader.readline()


access_token = ''  # Place auth code here
headers = {'Authorization': "Token " + access_token}

'''
data.csv is where the output of the data is stored

repos.csv is the result of a repo list generated using BigQuery
Steps to obtain this data:
1. Ensure you have a google account
2. Open https://console.cloud.google.com/bigquery. Create a new project to enable access to the query.
3. Open https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=github_repos to access the editor
4. Type the following into the editor
SELECT DISTINCT repo.name as repo FROM `githubarchive.day.2020*`
WHERE type = 'MemberEvent'

5. The program above will look at all github MemberEvent data from 2020 (starting on Jan 1). MemberEvents are recorded every time a new person becomes a contributor to a repo.
6. Once result is generated, click save result and export to google drive and download the csv file as repos.csv.
'''

counter = 1
with open('./../../data/raw/gith/data.csv', 'w', newline='') as output_csv, open('./../../data/raw/gith/repos.csv', 'r') as reader:
    w = csv.DictWriter(output_csv, ['repo', 'collabs', 'langs', 'rels', 'stargazers_count', 'forks_count', 'created_at',
                                    'pushed_at'])
    w.writeheader()
    repo = reader.readline()

    schedule.every(8).seconds.do(get_data, w=w)  # NOT schedule.every(1).seconds.do(getData, w=w)

    while repo != '':
        schedule.run_pending()
        time.sleep(1)
