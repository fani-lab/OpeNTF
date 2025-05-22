from huggingface_hub import HfApi, hf_hub_download, file_exists


'''
Obtain hf token:
Go to https://huggingface.co/settings/tokens and create a new token
'''
hf_api = HfApi(
    endpoint="https://huggingface.co/", # Can be a Private Hub endpoint.
    token="", # Token is not persisted on the machine.
)


# Upload as a folder
hf_api.upload_folder(
    folder_path="./output/gith/toy.repos.csv",
    path_in_repo="/output/gith/toy.repos.csv",
    repo_id="fani-lab/OpeNTF-test",
    repo_type="dataset"
)


# Upload individual files
hf_api.upload_file(
    path_or_fileobj="./output/gith/toy.repos.csv.mt10.ts2/indexes.pkl",
    path_in_repo="/output/gith/toy.repos.csv.mt10.ts2/indexes.pkl",
    repo_id="fani-lab/OpeNTF-test",
    repo_type="dataset"
)


# # Download a file from the hub
if(hf_api.file_exists(repo_id="fani-lab/OpeNTF-test", repo_type="dataset", filename="teams.pkl")):
    hf_api.hf_hub_download(repo_id="fani-lab/OpeNTF-test", repo_type="dataset", filename="output/gith/toy.repos.csv.mt10.ts2/indexes.pkl", local_dir='.')
else:
    print("File does not exist")
