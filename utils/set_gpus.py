import os
import torch
from utils.tprint import tprint

def set_gpus(gpus):
    """
    Set CUDA_VISIBLE_DEVICES environment variable based on available GPUs.
    
    Args:
        gpus: GPU indices to use (comma-separated string like "0" or "0,1")
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        tprint(f"{num_gpus} GPUs detected. Using GPU indices: {gpus} .")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif num_gpus == 1:
        tprint("Only one GPU detected. Using it (if CUDA is available).")
    else:
        tprint("No GPUs detected. Using CPU.") 