import csv, requests, traceback

access_token = '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'  # Place auth code here
headers = {'Authorization': "Token " + access_token}

with open('./../../data/raw/gith/data.csv', 'w', newline='') as output_csv, open('./../../data/raw/gith/repos.txt', 'r') as reader:
    w = csv.DictWriter(output_csv, ['repo', 'collabs', 'langs', 'rels'])
    w.writeheader()
    repo = reader.readline()
    while repo != '':
        repo = repo.rstrip('\n')
        try:
            collabs = requests.get(f'https://api.github.com/repos/{repo}/contributors', headers=headers).json()
            langs = requests.get(f'https://api.github.com/repos/{repo}/languages', headers=headers).json()
            rels = requests.get(f'https://api.github.com/repos/{repo}/releases', headers=headers).json()
            w.writerow({'repo': repo, 'collabs': collabs, 'langs': langs, 'rels': rels})
        except:
            print(f'Error occured for repo: {repo}')
            traceback.print_exc()
        repo = reader.readline()