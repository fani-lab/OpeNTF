#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to determine the default number of threads based on processing mode.
"""

from utils.tprint import tprint
from utils.parse_nthreads import parse_nthreads
from cmn_v3.helper_functions.import_gpu_libs import SELECTED_GPU_DEVICES


def get_default_threads(mode="cpu"):
    """
    Get the default number of threads/workers to use based on mode

    Args:
        mode: 'cpu' or 'gpu'

    Returns:
        int: Number of threads/workers to use
    """
    if mode == "gpu":
        # For GPU mode, use one worker per GPU device
        if SELECTED_GPU_DEVICES:
            return len(SELECTED_GPU_DEVICES)
        else:
            # Fallback to CPU mode if no GPUs are available
            return parse_nthreads()
    else:
        # For CPU mode, use the parse_nthreads function
        return parse_nthreads()
