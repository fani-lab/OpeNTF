import subprocess, sys, importlib, random, numpy, logging
log = logging.getLogger(__name__)
from omegaconf import OmegaConf
import re
from itertools import chain
from importlib.metadata import version

pkg_req_dict = {}

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

def is_version_equal(version_inst: str, version_req: str) -> bool:
    """
    Compare two version strings.
    Returns True if they are equal, False otherwise.
    version_inst: installed version string
    version_req: version string from requirements file
    """

    # Check for trailing .* in version_req
    if version_req.endswith('.*'): version_req = version_req[:-2]  + version_inst[version_inst.rfind('.'):]
        
    return version_inst == version_req

def install_pkg(pkg_name):
    log.info(f'Installing {pkg_name}...')
    process = subprocess.run([sys.executable, '-m', 'pip', 'install'] + pkg_req_dict[pkg_name][0].split(), text=True, capture_output=True)#-m makes the pip to work as module inside env, not the system pip!
    log.info(process.stdout)
    if process.returncode != 0: raise ImportError(f'Failed to install package: {pkg_name}\n{process.stderr}')

def reinstall_pkg(pkg_name):
    log.info(f'Uninstalling {pkg_name}...')
    process = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg_name], text=True, capture_output=True)#-m makes the pip to work as module inside env, not the system pip!
    log.info(process.stdout)
    if process.returncode != 0: raise ImportError(f'Failed to uninstall package: {pkg_name}\n{process.stderr}')
    install_pkg(pkg_name)


def install_import_2(pkg_name, import_path=None, from_module=None):
    """
    pkg_name: the name of the package to install (e.g., "beautifulsoup4)
    import_path: full module path to import (e.g., "bs4.BeautifulSoup")
    from_module: if set, return only the object from module (e.g., BeautifulSoup class)
    """
    import_path = import_path or pkg_name
    try: 
        module = importlib.import_module(import_path)
        if(not is_version_equal(version(pkg_name), pkg_req_dict[pkg_name][1])):
            log.info(f"Version mismatch detected. {pkg_name} version {version(pkg_name)} is installed, but {pkg_req_dict[pkg_name][1]} is required.")
            reinstall_pkg(pkg_name)
            module = importlib.import_module(import_path)

    except ImportError:
       log.info(f'{import_path} not found.')
       install_pkg(pkg_name)
       module = importlib.import_module(import_path)

    if from_module: return getattr(module, from_module)
    return module

def set_seed(seed, torch=None):
    if seed is None: return
    random.seed(seed)
    numpy.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        #torch.use_deterministic_algorithms(True) #RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation .torch.nn.functional.leaky_relu is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

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

def generate_req_dict(req_file):
    """
    Generates a dictionary of packages from a requirements file.
    The keys are package names, and the values are tuples of (line, version).
    req_file: path to the requirements file
    """
    global pkg_req_dict
    if pkg_req_dict: return pkg_req_dict

    
    def extract_package_info_from_line(line):
        """
        line: a string from the requirements file that starts with "#$"
        """
        line = line[2:] # Remove the "#$"
        line = line.split("#")[0]  # Remove comments
        line = line.strip() # Remove leading and trailing whitespace


        package_name = "([-A-Za-z0-9_\.]+)"
        comp = "(==|!=|<=|>=|<|>|~=|===)"
        ver_num = "([0-9]+[0-9\.\*]*)"

        out = []

        for pkg in re.findall(f"{package_name}[\s]*{comp}[\s]*{ver_num}", line):
            out.append((pkg[0], (line, pkg[2])))

        return out # [(package_name, (line, ver_num)), ...])]
    
    with open(req_file, 'r') as f:
        filtered_lines = dict(chain.from_iterable(map(lambda line: extract_package_info_from_line(line), filter(lambda x: x.startswith("#$"), f.readlines()))))
        pkg_req_dict = filtered_lines
        log.info(f'Extracted {len(pkg_req_dict)} packages from {req_file}')
        log.info(f'Requirement Dictionary: {pkg_req_dict}')

hf_token = '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' # this is a read-only token
def get_from_hf(repo_type, filename) -> bool:
    hf_api = install_import('huggingface-hub==0.33.0', 'huggingface_hub', 'HfApi')()
    repo_id = 'fani-lab/OpeNTF'
    log.info(f"Downloading {filename} from https://huggingface.co/{repo_type}/{repo_id} ...")
    # if the file is public, token=hf_token is ignored
    try: return hf_api.hf_hub_download(repo_id='fani-lab/OpeNTF', repo_type=repo_type, filename=filename, local_dir='../', force_download=True)
    except Exception as e: log.error(f'Error downloading {filename} from {repo_id}! {e}'); return None