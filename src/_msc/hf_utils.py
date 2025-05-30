import os 
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

'''
Obtain hf token:
1. Go to https://huggingface.co/settings/tokens
2, Create a new token and got to section "Repositories Permission"
3. Search for the "datasets/fani-lab/OpeNTF-test" and then check the following permissions:
    - Read access to contents of selected repos
    - Write access to contents/settings of selected repos
4.In src/.env, add the token to the line with HF_API_TOKEN
'''
hf_api = HfApi(token=os.getenv("HF_API_TOKEN"))


# Upload as a folder
# hf_api.upload_folder(
#     folder_path="./output/gith/toy.repos.csv",
#     path_in_repo="/output/gith/toy.repos.csv",
#     repo_id="fani-lab/OpeNTF-test",
#     repo_type="dataset"
# )


# Upload individual files
# hf_api.upload_file(
#     path_or_fileobj="./output/gith/toy.repos.csv.mt10.ts2/teamsvecs.pkl",
#     path_in_repo="/output/gith/toy.repos.csv.mt10.ts2/teamsvecs.pkl",
#     repo_id="fani-lab/OpeNTF-test",
#     repo_type="dataset"
# )


# Download a file from the hub
# if(hf_api.file_exists(repo_id="fani-lab/OpeNTF-test", repo_type="dataset", filename="output/gith/toy.repos.csv.mt10.ts2/indexes.pkl")):
#     hf_api.hf_hub_download(repo_id="fani-lab/OpeNTF-test", repo_type="dataset", filename="output/gith/toy.repos.csv.mt10.ts2/indexes.pkl", local_dir='.')
# else:
#     print("File does not exist")
