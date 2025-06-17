import subprocess, sys, importlib, random, numpy
import logging
log = logging.getLogger(__name__)
from omegaconf import OmegaConf
def install_import(install_name, import_path=None, from_module=None):
    """
    install_name: name used in pip install, may be different from the import name/path
    import_path: full module path to import (e.g., "bs4.BeautifulSoup")
    from_module: if set, return only the object from module (e.g., BeautifulSoup class)
    #samples
    > install_import('hydra-core==1.3.2', 'hydra')
    Importing a submodule/class/function: from bs4 import BeautifulSoup
    > BeautifulSoup = install_and_import('beautifulsoup4', 'bs4', 'BeautifulSoup')
    > soup = BeautifulSoup('<html><body><p>Hello</p></body></html>', 'html.parser')
    > print(soup.p.text)  # -> "Hello"
    """
    import_path = import_path or install_name
    try: module = importlib.import_module(import_path)
    except ImportError:
        log.info(f'{import_path} not found. Installing {install_name}...')
        process = subprocess.run([sys.executable, '-m', 'pip', 'install'] + install_name.split(), text=True, capture_output=True)#-m makes the pip to work as module inside env, not the system pip!
        log.info(process.stdout)
        #if process.stderr: log.info(process.stderr)
        if process.returncode != 0: raise ImportError(f'Failed to install package: {install_name}\n{process.stderr}')
        module = importlib.import_module(import_path)

    if from_module: return getattr(module, from_module)
    return module

def set_seed(seed, torch=None):
    if seed is None: return
    random.seed(seed)
    numpy.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if multiple GPUs
            torch.backends.cudnn.deterministic = True # in cuDNN
            torch.backends.cudnn.benchmark = False

def cfg2str(cfg): return '.'.join([f'{k}{v}' for k, v in OmegaConf.to_container(cfg, resolve=True).items()])

def str2cfg(s): #dot seperated kv, e.g., x1.y2.z3 --> x:1 y:2 z:3
    items = s.split(".")
    config = {}
    for item in items:
        key = ''.join(filter(str.isalpha, item))
        value = ''.join(filter(str.isdigit, item))
        config[key] = int(value) if value.isdigit() else value
    return OmegaConf.create(config)

textcolor = {
    'blue':   '\033[94m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'magenta':'\033[95m',
    'reset':  '\033[0m'
}
hf_token = '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'
def get_from_hf(repo_type, filename) -> bool:
    hf_api = install_import('huggingface-hub==0.33.0', 'huggingface_hub', 'HfApi')()
    repo_id = 'fani-lab/OpeNTF'
    log.info(f"Downloading {filename} from https://huggingface.co/{repo_type}/{repo_id} ...")
    # if the file is public, token=hf_token is ignored
    try: return hf_api.hf_hub_download(repo_id='fani-lab/OpeNTF', repo_type=repo_type, filename=filename, local_dir='../', force_download=True, token=hf_token)
    except Exception as e: log.error(f'Error downloading {filename} from {repo_id}! {e}'); return None