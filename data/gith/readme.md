`links.csv` is the result of a repo list generated using `BigQuery`

Steps to obtain this data:
1. Ensure you have a Google account
2. Open https://console.cloud.google.com/bigquery. Create a new project to enable access to the query.
3. Open https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=github_repos to access the editor
4. Type the following into the editor

```
SELECT DISTINCT repo.name as repo FROM `githubarchive.day.2020*`
WHERE type = 'MemberEvent'
```

5. The sql query above will look at all github MemberEvent data from 2020 (starting on Jan 1). MemberEvents are recorded every time a new person becomes a contributor to a repo.
7. Once result is generated, click save result and export to google drive and download the csv file as links.csv.
8. Then, run [`github_crawler.py`](../../src/_msc/github_crawler.py) to crawl the detail information of each repo link, which saves `repos.csv` in the following file structure:

```
repo, collabs
fani-lab/OpeNTF, [
                {'login': 'hosseinfani',
                'id': 8619934,
                'node_id': 'MDQ6VXNlcjg2MTk5MzQ=',                        
                'avatar_url': 'https://avatars.githubusercontent.com/u/8619934?v=4',
                'gravatar_id': '', 
                'url': 'https://api.github.com/users/hosseinfani',
                'html_url': 'https://github.com/hosseinfani',
                'followers_url': 'https://api.github.com/users/hosseinfani/followers',
                'following_url': 'https://api.github.com/users/hosseinfani/following{/other_user}',
                'gists_url': 'https://api.github.com/users/hosseinfani/gists{/gist_id}',
                'starred_url': 'https://api.github.com/users/hosseinfani/starred{/owner}{/repo}',
                'subscriptions_url': 'https://api.github.com/users/hosseinfani/subscriptions',
                'organizations_url': 'https://api.github.com/users/hosseinfani/orgs',
                'repos_url': 'https://api.github.com/users/hosseinfani/repos',
                'events_url': 'https://api.github.com/users/hosseinfani/events{/privacy}',
                'received_events_url': 'https://api.github.com/users/hosseinfani/received_events',
                'type': 'User',
                'site_admin': False,
                'contributions': 300},
                ...
                ]
```

Stats: 

|Stat| Value|
|-----|------|
|#Repos (teams)| 661,335|
|#Unique Contributors (members) |1,444,480|
|#Unique Programming Language (skills)|20|
|Avg #Contributors per Repo|5.5345278867744785|
|Avg #Programmng Language per Repo|1.3723967429517567|
|Avg #Repo per Contributor|2.533906319229065|
|Avg #Programming Language per Contributor|2.1095840717766947|
|#Repo w/o Programming Language |?|
|#Repo w/ Single Contributor|0|
|#Repo w/ Single Programming Language |342,950|
