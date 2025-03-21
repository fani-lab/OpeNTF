#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for determining the optimal number of threads
to use for parallel processing based on settings in param.py and system capabilities.


Available values for "nthreads":
-------------------------------
1. Percentage of threads (float between 0.0 and 1.0):
   "nthreads": 0.5  # Use 50% of available CPU threads
   "nthreads": 0.75  # Use 75% of available CPU threads

2. Specific number of threads (positive integer):
   "nthreads": 16  # Use exactly 16 threads
   "nthreads": 32  # Use exactly 32 threads

3. Use all available threads (any of these):
   "nthreads": 0    # All threads
   "nthreads": -1   # All threads (common convention)
   "nthreads": 1.0  # All threads (100%)
   "nthreads": 1.5  # All threads (values > 1.0 use all threads)

For large systems (e.g., with 112 cores/224 threads), using a percentage like 0.5
is recommended to avoid system overload while still getting good performance.
"""

import multiprocessing
from utils.tprint import tprint
import os


def parse_nthreads(settings=None):
    """
    Get the optimal number of threads to use based on settings in param.py.

    Behavior:
    - If nthreads is a float between 0.0 and 1.0, treat it as a percentage of total threads
    - If nthreads is a positive integer < total_threads, use that specific number of threads
    - If nthreads is negative, 0, 0.0, 1.0, or any decimal > 1.0, use all available threads

    Examples:
    - nthreads = 0.5 -> 50% of available threads (112 on a 224-thread system)
    - nthreads = 16 -> exactly 16 threads (regardless of system capabilities)
    - nthreads = -1 -> all available threads (224 on a 224-thread system)
    - nthreads = 0 -> all available threads (224 on a 224-thread system)
    - nthreads = 1.5 -> all available threads (224 on a 224-thread system)

    Returns:
        int: Number of threads to use for parallel processing
    """
    # Get total available threads
    total_threads = multiprocessing.cpu_count()

    # Initialize nthreads with a default value of 0 (use all threads)
    nthreads = 0

    # Handle default case
    if settings is None:
        try:
            # Try to import from param.py
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from src import param

            nthreads = param.settings["data"]["processing"].get("nthreads", 0)
        except (ImportError, AttributeError, KeyError) as e:
            # If import fails or param doesn't have the expected structure, use default
            tprint(
                f"Could not load nthreads from param.py, using all available threads: {total_threads}"
            )
    elif isinstance(settings, str):
        # If settings is a string path, use default value
        tprint(
            f"Settings provided as string path, using all available threads: {total_threads}"
        )
    else:
        # Extract nthreads from settings dictionary
        try:
            nthreads = settings["data"]["processing"].get("nthreads", 0)
        except (TypeError, KeyError):
            # If settings does not have the expected structure, use default
            tprint(
                f"Invalid settings format, using all available threads: {total_threads}"
            )

    # Case 1: nthreads is a float between 0.0 and 1.0 exclusive (percentage)
    if isinstance(nthreads, float) and 0.0 < nthreads < 1.0:
        thread_count = max(int(total_threads * nthreads), 1)
        tprint(
            f"Using {nthreads:.1%} of available threads: {thread_count} of {total_threads}"
        )
        return thread_count

    # Case 2: nthreads is a positive integer less than total_threads (specific number)
    elif isinstance(nthreads, int) and 0 < nthreads < total_threads:
        tprint(f"Using {nthreads} threads (of {total_threads} available)")
        return nthreads

    # Case 3: All other cases - use all threads
    # - Negative values (e.g., -1)
    # - Zero (0, 0.0)
    # - One (1.0 as float)
    # - Values >= total_threads
    # - Floats >= 1.0
    else:
        tprint(f"Using all available threads: {total_threads}")
        return total_threads


if __name__ == "__main__":
    # Simple test when run directly
    thread_count = parse_nthreads()
    print(
        f"Thread count: {thread_count} of {multiprocessing.cpu_count()} available threads"
    )
