import logging, os
from huggingface_hub import HfApi
from dotenv import load_dotenv

log = logging.getLogger(__name__)
load_dotenv()

class HFClient(object):
    def __init__(self, repo_id: str, repo_type: str):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.hf_api = HfApi(token=os.getenv("HF_API_TOKEN"))

    def download_file(self, filename: str) -> bool:
        try:
            if(self.file_exists(filename)):
                log.info(f"Downloading file from hf: {filename} ...")
                self.hf_api.hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type, filename=filename, local_dir='../')
                return True
            
            log.info(f"File {filename} does not exist in the repository {self.repo_id}.")
            return False
        except Exception as e:
            log.error(f"Error downloading file: {e}")
            return False
    
    def file_exists(self, filename: str) -> bool:
        try:
            return self.hf_api.file_exists(repo_id=self.repo_id, repo_type=self.repo_type, filename=filename)
        except Exception as e:
            log.error(f"Error checking file existence: {e}")
            return False