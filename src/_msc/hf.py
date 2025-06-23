import os

import pkgmgr as opentf

'''
Obtain hf token:
1. Go to https://huggingface.co/settings/tokens
2, Create a new token and got to section "Repositories Permission"
3. Search for the "datasets/fani-lab/OpeNTF" and then check the following permissions:
    - Read access to contents of selected repos
    - Write access to contents/settings of selected repos
4.In src/.env, add the token to the line with HF_API_TOKEN
'''

HfApi = opentf.install_import('huggingface-hub==0.33.0', 'huggingface_hub', 'HfApi')
hf_api = HfApi(token=opentf.hf_token)


# Upload as a folder
# hf_api.upload_folder(
#     folder_path="./output/gith/toy.repos.csv",
#     path_in_repo="/output/gith/toy.repos.csv",
#     repo_id="fani-lab/OpeNTF",
#     repo_type="dataset"
# )


# Upload teamsvecs from toy dblp
hf_api.upload_file(
    path_or_fileobj="../output/dblp/toy.dblp.v12.json/teamsvecs.pkl",
    path_in_repo="/output/dblp/toy.dblp.v12.json/teamsvecs.pkl",
    repo_id="fani-lab/OpeNTF",
    repo_type="dataset"
)

# Upload teamsvecs from toy gith
hf_api.upload_file(
    path_or_fileobj="../output/gith/toy.repos.csv/teamsvecs.pkl",
    path_in_repo="/output/gith/toy.repos.csv/teamsvecs.pkl",
    repo_id="fani-lab/OpeNTF",
    repo_type="dataset"
)


# Upload teamsvecs from toy imdb
hf_api.upload_file(
    path_or_fileobj="../output/imdb/toy.title.basics.tsv/teamsvecs.pkl",
    path_in_repo="/output/imdb/toy.title.basics.tsv/teamsvecs.pkl",
    repo_id="fani-lab/OpeNTF",
    repo_type="dataset"
)


# Upload teamsvecs from toy uspt
hf_api.upload_file(
    path_or_fileobj="../output/uspt/toy.patent.tsv/teamsvecs.pkl",
    path_in_repo="/output/uspt/toy.patent.tsv/teamsvecs.pkl",
    repo_id="fani-lab/OpeNTF",
    repo_type="dataset"
)




# hf_api.delete_folder(
#     path_in_repo="/output/",
#     repo_id="fani-lab/OpeNTF",
#     repo_type="dataset"
# )

# hf_api.delete_file(
#     path_in_repo="teamsvecs.pkl",
#     repo_id="fani-lab/OpeNTF",
#     repo_type="dataset"
# )


# Download a file from the hub
# if(hf_api.file_exists(repo_id="fani-lab/OpeNTF", repo_type="dataset", filename="output/gith/toy.repos.csv.mt10.ts2/indexes.pkl")):
#     hf_api.hf_hub_download(repo_id="fani-lab/OpeNTF", repo_type="dataset", filename="output/gith/toy.repos.csv.mt10.ts2/indexes.pkl", local_dir='.')
# else:
#     print("File does not exist")
