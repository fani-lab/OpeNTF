import subprocess, sys, importlib
import logging
log = logging.getLogger(__name__)
def install_import(install_name, import_path=None, from_module=None):
    """
    install_name: name used in pip install, may be different from the import name/path
    import_path: full module path to import (e.g., "bs4.BeautifulSoup")
    from_module: if set, return only the object from module (e.g., BeautifulSoup class)
    """
    import_path = import_path or install_name
    try: module = importlib.import_module(import_path)
    except ImportError:
        log.info(f'{import_path} not found. Installing {install_name}...')
        process = subprocess.run([sys.executable, "-m", "pip", "install", install_name], text=True, capture_output=True)#-m makes the pip to work as module inside env, not the system pip!
        log.info(process.stdout)
        if process.stderr: log.info(process.stderr)
        if process.returncode != 0: raise ImportError(f"Failed to install package: {install_name}")
        module = importlib.import_module(import_path)

    if from_module: return getattr(module, from_module)
    return module

# #samples
# install_import('hydra-core==1.3.2', 'hydra')
# # Importing a submodule/class/function: from bs4 import BeautifulSoup
# BeautifulSoup = install_and_import('beautifulsoup4', 'bs4', 'BeautifulSoup')
# soup = BeautifulSoup('<html><body><p>Hello</p></body></html>', 'html.parser')
# print(soup.p.text)  # -> "Hello"