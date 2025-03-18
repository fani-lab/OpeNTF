#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate reports and visualizations for team data.

This module can use GPU acceleration to significantly speed up processing.
To use GPU acceleration:
1. Make sure CuPy is installed: pip install cupy-cuda11x (match cuda version to your system)
2. Run with the -mode gpu option for automatic GPU selection
3. For specific GPU selection, use -mode gpu=0 or -mode gpu=0,1,2 for multiple GPUs
4. Use -mode gpu=all to use all available GPUs

Note: If you've used the -gpus parameter in main.py (e.g., -gpus 2), this script will only
see the GPUs that are visible to the system at this point. For example, if you specified
-gpus 2 in main.py, then this script will only see one GPU (index 0) which is actually
GPU index 2 from the system's perspective. The first available GPU in this script will
correspond to whatever GPU was made visible by the earlier gpus parameter.
"""

import pickle
import os
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt
import matplotlib
import datetime  # Add explicit import for datetime

matplotlib.use("Agg")  # Use non-interactive backend
import pandas as pd
from collections import Counter
import seaborn as sns
from scipy.sparse import csr_matrix
import matplotlib.cm as cm
from scipy import stats
from datetime import datetime
import time
import multiprocessing
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
import math

# Try to import pytz, but provide a fallback if not available
try:
    import pytz

    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("Warning: pytz not available. Using UTC timezone for reports.")

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint, get_est_time

# Import the refactored functions and constants
from cmn_v3.helper_functions.import_gpu_libs import (
    CUPY_AVAILABLE,
    SELECTED_GPU_DEVICES,
    import_gpu_libs,
    parse_gpu_string,
)
from cmn_v3.helper_functions.get_gpu_device import get_gpu_device
from cmn_v3.helper_functions.get_default_threads import get_default_threads
from cmn_v3.helper_functions.get_nthreads import get_nthreads

# Default number of threads for parallel processing
# We'll reduce this if GPU mode is active to prevent memory contention
DEFAULT_THREADS = get_nthreads()


def get_default_threads(mode_str="cpu"):
    """
    Get the default number of threads based on the execution mode.

    Args:
        mode_str: The execution mode (cpu or gpu)

    Returns:
        Number of threads to use for parallel processing
    """
    # If using CPU mode, use all available threads
    if mode_str == "cpu" or not mode_str.startswith("gpu"):
        return DEFAULT_THREADS

    # If using GPU mode, reduce thread count to prevent memory contention
    # Use just 1/4 of available threads when using GPU
    gpu_threads = max(4, DEFAULT_THREADS // 4)
    tprint(
        f"Using reduced thread count ({gpu_threads}) for GPU mode to prevent memory contention"
    )
    return gpu_threads


def analyze_teams(teamsvecs, n_jobs=None):
    """
    Analyze teams to extract statistics

    Args:
        teamsvecs: Dictionary containing team vectors
        n_jobs: Number of parallel jobs to use

    Returns:
        Dictionary of team statistics
    """
    if n_jobs is None:
        # Get the appropriate thread count based on whether GPU is available
        if CUPY_AVAILABLE and SELECTED_GPU_DEVICES:
            n_jobs = get_default_threads("gpu")
        else:
            n_jobs = DEFAULT_THREADS

    tprint(f"Analyzing team details in parallel using {n_jobs} jobs...")

    # Extract matrices from teamsvecs
    skill_matrix = teamsvecs["skill"]
    member_matrix = teamsvecs["member"]

    n_teams = skill_matrix.shape[0]

    # Process in batches using multiprocessing
    batch_size = max(1, min(10000, n_teams // (n_jobs * 2)))  # Smaller batches with GPU
    n_batches = math.ceil(n_teams / batch_size)

    tprint(
        f"Processing {n_teams} teams in {n_batches} batches using {n_jobs} parallel jobs..."
    )

    # Prepare batches
    batches = [
        (i * batch_size, min((i + 1) * batch_size, n_teams)) for i in range(n_batches)
    ]

    # Process batches in parallel
    results = []

    # Check if GPU processing is feasible for this dataset size
    use_gpu = CUPY_AVAILABLE and SELECTED_GPU_DEVICES
    if use_gpu:
        # Check if dataset is too large for GPU processing
        try:
            # Get the current GPU device's free memory
            import cupy as cp

            device_id = (
                SELECTED_GPU_DEVICES[0] if isinstance(SELECTED_GPU_DEVICES, list) else 0
            )
            with cp.cuda.Device(device_id):
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                free_gb = free_mem / (1024**3)

                # Estimate memory required for processing
                member_cols = member_matrix.shape[1]
                skill_cols = skill_matrix.shape[1]
                largest_batch_size = max(end - start for start, end in batches)
                est_mem_per_batch = (
                    largest_batch_size * (member_cols + skill_cols) * 4 / (1024**3)
                )  # in GB

                if (
                    est_mem_per_batch > free_gb * 0.4
                ):  # If batch needs more than 40% of free memory
                    tprint(
                        f"Warning: Individual batches may be too large for GPU memory"
                    )
                    tprint(
                        f"Estimated memory per batch: {est_mem_per_batch:.2f}GB, Free memory: {free_gb:.2f}GB"
                    )
                    tprint(f"Reducing batch size and increasing number of batches")

                    # Recalculate with smaller batches
                    max_rows_per_batch = int(
                        (free_gb * 0.4 * 1024**3) / ((member_cols + skill_cols) * 4)
                    )
                    batch_size = max(
                        1, min(max_rows_per_batch, 5000)
                    )  # Cap at 5000 rows per batch
                    n_batches = math.ceil(n_teams / batch_size)

                    tprint(
                        f"Adjusted to {n_batches} batches with approximately {batch_size} rows each"
                    )

                    # Redefine batches
                    batches = [
                        (i * batch_size, min((i + 1) * batch_size, n_teams))
                        for i in range(n_batches)
                    ]
        except Exception as e:
            tprint(f"Error estimating GPU memory requirements: {str(e)}")
            # Continue with original batches

    # Process batches
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        futures = [
            executor.submit(
                process_team_batch, start_idx, end_idx, skill_matrix, member_matrix
            )
            for start_idx, end_idx in batches
        ]

        # Collect results as they complete
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)

    tprint(f"Combining results from parallel batch processing...")

    # Combine results from all batches
    all_skills_per_team = []
    all_experts_per_team = []
    all_skill_indices = []
    all_expert_indices = []
    zero_skill_teams = 0
    zero_expert_teams = 0
    max_skills = 0
    min_skills = float("inf")
    max_experts = 0
    min_experts = float("inf")
    dup_teams = 0
    seen_configs = {}
    all_unique_skills = set()
    all_unique_experts = set()

    for batch_result in results:
        (
            skills_per_team_batch,
            experts_per_team_batch,
            skill_indices_per_team_batch,
            expert_indices_per_team_batch,
            zero_skill_teams_batch,
            zero_expert_teams_batch,
            max_skills_batch,
            min_skills_batch,
            max_experts_batch,
            min_experts_batch,
            dup_teams_batch,
            seen_configs_batch,
            unique_skills_batch,
            unique_experts_batch,
        ) = batch_result

        # Combine team statistics
        all_skills_per_team.extend(skills_per_team_batch)
        all_experts_per_team.extend(experts_per_team_batch)
        all_skill_indices.extend(skill_indices_per_team_batch)
        all_expert_indices.extend(expert_indices_per_team_batch)

        # Combine zero teams count
        zero_skill_teams += zero_skill_teams_batch
        zero_expert_teams += zero_expert_teams_batch

        # Update min/max values
        if min_skills_batch < min_skills:
            min_skills = min_skills_batch
        if max_skills_batch > max_skills:
            max_skills = max_skills_batch
        if min_experts_batch < min_experts:
            min_experts = min_experts_batch
        if max_experts_batch > max_experts:
            max_experts = max_experts_batch

        # Combine duplicate count
        dup_teams += dup_teams_batch
        seen_configs.update(seen_configs_batch)

        # Update unique skills and experts
        all_unique_skills.update(unique_skills_batch)
        all_unique_experts.update(unique_experts_batch)

    # Handle case when no valid min values were found
    if min_skills == float("inf"):
        min_skills = 0
    if min_experts == float("inf"):
        min_experts = 0

    # Calculate final statistics
    tprint(
        f"Actual unique skills used in dataset: {len(all_unique_skills)} out of {skill_matrix.shape[1]}"
    )
    tprint(
        f"Actual unique experts used in dataset: {len(all_unique_experts)} out of {member_matrix.shape[1]}"
    )

    # Count team participation per expert
    tprint(f"Calculating team participation per expert...")
    expert_participation = {}
    for team_expert_indices in all_expert_indices:
        for expert_idx in team_expert_indices:
            if expert_idx in expert_participation:
                expert_participation[expert_idx] += 1
            else:
                expert_participation[expert_idx] = 1

    # Find max and min participation
    team_count_per_expert = Counter(expert_participation.values())
    min_participation = (
        min(expert_participation.values()) if expert_participation else 0
    )
    max_participation = (
        max(expert_participation.values()) if expert_participation else 0
    )

    # Count experts with min/max participation
    min_participation_count = (
        team_count_per_expert[min_participation]
        if min_participation in team_count_per_expert
        else 0
    )
    max_participation_count = (
        team_count_per_expert[max_participation]
        if max_participation in team_count_per_expert
        else 0
    )

    tprint(
        f"Found {min_participation_count} experts with min participation ({min_participation} teams)"
    )
    tprint(
        f"Found {max_participation_count} experts with max participation ({max_participation} teams)"
    )

    # Count unique team configurations
    tprint(f"Calculating team statistics...")
    unique_team_configs = len(seen_configs)
    tprint(
        f"Found {unique_team_configs} unique team configurations out of {n_teams} teams"
    )

    # Prepare statistics dictionary
    stats = {
        "n_teams": n_teams,
        "n_unique_team_configs": unique_team_configs,
        "skills_per_team": all_skills_per_team,
        "experts_per_team": all_experts_per_team,
        "zero_skill_teams": zero_skill_teams,
        "zero_expert_teams": zero_expert_teams,
        "min_skills": min_skills,
        "max_skills": max_skills,
        "min_experts": min_experts,
        "max_experts": max_experts,
        "expert_participation": expert_participation,
        "min_participation": min_participation,
        "max_participation": max_participation,
        "min_participation_count": min_participation_count,
        "max_participation_count": max_participation_count,
        "unique_skills": len(all_unique_skills),
        "unique_experts": len(all_unique_experts),
        "all_skill_indices": all_skill_indices,
        "all_expert_indices": all_expert_indices,
        "dup_teams": dup_teams,
    }

    return stats


def process_team_batch(start_idx, end_idx, skill_matrix, member_matrix):
    """
    Process a batch of teams in parallel

    Args:
        start_idx: Start index for this batch
        end_idx: End index for this batch
        skill_matrix: Matrix of team skills
        member_matrix: Matrix of team members

    Returns:
        Tuple containing batch statistics
    """
    # Detect dimensions to help with processing decisions
    batch_size = end_idx - start_idx
    member_cols = member_matrix.shape[1]
    skill_cols = skill_matrix.shape[1]

    # Check if the matrix is extremely high dimensional
    # For very large member matrices, skip GPU processing entirely
    if member_cols > 500000:
        # For extremely large member matrices, always use CPU
        return _process_team_batch_cpu(start_idx, end_idx, skill_matrix, member_matrix)

    # Check if GPU acceleration can be used
    try:
        gpu_device = get_gpu_device()
        if gpu_device is None:
            # If no GPU available, use CPU implementation without warning
            return _process_team_batch_cpu(
                start_idx, end_idx, skill_matrix, member_matrix
            )

        # Use GPU acceleration with memory management
        import cupy as cp

        # Calculate batch size based on matrix dimensions to prevent OOM errors
        # Reduce max_elements for matrices with many columns to avoid memory issues
        max_elements = min(
            500_000_000, 200_000_000 if member_cols > 100000 else 500_000_000
        )

        if batch_size * (member_cols + skill_cols) > max_elements:
            # Skip GPU processing for large batches without repetitive error messages
            if batch_size > 1000 or member_cols > 50000:
                tprint(
                    f"Large batch detected ({batch_size} rows, {member_cols} member cols, {skill_cols} skill cols)"
                )
                return _process_team_batch_cpu(
                    start_idx, end_idx, skill_matrix, member_matrix
                )

        try:
            with gpu_device:
                # Transfer data to GPU
                batch_skill_rows = skill_matrix[start_idx:end_idx]
                batch_member_rows = member_matrix[start_idx:end_idx]

                # Check memory availability before conversion
                try:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    free_mem_gb = free_mem / (1024**3)
                    estimated_mem_needed = (
                        batch_size * (member_cols + skill_cols) * 4 / (1024**3)
                    )  # in GB

                    if (
                        estimated_mem_needed > free_mem_gb * 0.8
                    ):  # Only use 80% of available memory
                        tprint(
                            f"Insufficient GPU memory: need {estimated_mem_needed:.2f}GB, have {free_mem_gb:.2f}GB free"
                        )
                        return _process_team_batch_cpu(
                            start_idx, end_idx, skill_matrix, member_matrix
                        )
                except Exception as mem_error:
                    # If memory check fails, fall back to CPU
                    tprint(f"Error checking GPU memory: {str(mem_error)}")
                    return _process_team_batch_cpu(
                        start_idx, end_idx, skill_matrix, member_matrix
                    )

                # Convert to GPU arrays
                try:
                    if hasattr(batch_skill_rows, "toarray"):
                        skill_gpu = cp.array(batch_skill_rows.toarray())
                    else:
                        skill_gpu = cp.array(batch_skill_rows)

                    if hasattr(batch_member_rows, "toarray"):
                        member_gpu = cp.array(batch_member_rows.toarray())
                    else:
                        member_gpu = cp.array(batch_member_rows)
                except cp.cuda.memory.OutOfMemoryError:
                    # Free memory and fall back to CPU
                    cp.get_default_memory_pool().free_all_blocks()
                    return _process_team_batch_cpu(
                        start_idx, end_idx, skill_matrix, member_matrix
                    )
                except Exception as array_error:
                    # Handle other array conversion errors
                    tprint(f"Error converting data to GPU arrays: {str(array_error)}")
                    cp.get_default_memory_pool().free_all_blocks()
                    return _process_team_batch_cpu(
                        start_idx, end_idx, skill_matrix, member_matrix
                    )

                # Calculate skills and experts per team using sum
                skills_per_team_gpu = cp.sum(skill_gpu > 0, axis=1)
                experts_per_team_gpu = cp.sum(member_gpu > 0, axis=1)

                # Transfer results back to CPU
                skills_per_team_batch = cp.asnumpy(skills_per_team_gpu).tolist()
                experts_per_team_batch = cp.asnumpy(experts_per_team_gpu).tolist()

                # Calculate zero skills and experts teams
                zero_skill_teams_batch = int(cp.sum(skills_per_team_gpu == 0).item())
                zero_expert_teams_batch = int(cp.sum(experts_per_team_gpu == 0).item())

                # Calculate min/max values
                if len(skills_per_team_batch) > 0:
                    # Filter out zero values for min calculation if any non-zero values exist
                    non_zero_skills = skills_per_team_gpu[skills_per_team_gpu > 0]
                    if len(non_zero_skills) > 0:
                        min_skills_batch = int(cp.min(non_zero_skills).item())
                    else:
                        min_skills_batch = 0
                    max_skills_batch = int(cp.max(skills_per_team_gpu).item())

                    non_zero_experts = experts_per_team_gpu[experts_per_team_gpu > 0]
                    if len(non_zero_experts) > 0:
                        min_experts_batch = int(cp.min(non_zero_experts).item())
                    else:
                        min_experts_batch = 0
                    max_experts_batch = int(cp.max(experts_per_team_gpu).item())
                else:
                    min_skills_batch = float("inf")
                    max_skills_batch = 0
                    min_experts_batch = float("inf")
                    max_experts_batch = 0

                # Free GPU memory before CPU operations
                del skills_per_team_gpu
                del experts_per_team_gpu
                del skill_gpu
                del member_gpu
                cp.get_default_memory_pool().free_all_blocks()

                # Need to process on CPU for the remaining operations
                # These operations involve creating tuples which is harder to do on GPU
                skill_indices_per_team_batch = []
                expert_indices_per_team_batch = []
                seen_configs_batch = {}
                dup_teams_batch = 0

                # Process each team to get skill and expert indices
                batch_skill_rows = skill_matrix[start_idx:end_idx]
                batch_member_rows = member_matrix[start_idx:end_idx]

                for i in range(end_idx - start_idx):
                    skill_row = batch_skill_rows[i]
                    member_row = batch_member_rows[i]

                    skill_indices = tuple(skill_row.nonzero()[1])
                    expert_indices = tuple(member_row.nonzero()[1])

                    # Store skill and expert indices
                    skill_indices_per_team_batch.append(skill_indices)
                    expert_indices_per_team_batch.append(expert_indices)

                    # Track team configuration for duplicate detection
                    team_config = (skill_indices, expert_indices)
                    team_idx = i + start_idx

                    if team_config in seen_configs_batch:
                        dup_teams_batch += 1
                    else:
                        seen_configs_batch[team_config] = team_idx

                # Sets to track unique skill and expert IDs used in this batch
                unique_skills_batch = set()
                unique_experts_batch = set()

                for skill_indices in skill_indices_per_team_batch:
                    unique_skills_batch.update(skill_indices)

                for expert_indices in expert_indices_per_team_batch:
                    unique_experts_batch.update(expert_indices)

                return (
                    skills_per_team_batch,
                    experts_per_team_batch,
                    skill_indices_per_team_batch,
                    expert_indices_per_team_batch,
                    zero_skill_teams_batch,
                    zero_expert_teams_batch,
                    max_skills_batch,
                    min_skills_batch,
                    max_experts_batch,
                    min_experts_batch,
                    dup_teams_batch,
                    seen_configs_batch,
                    unique_skills_batch,
                    unique_experts_batch,
                )
        except Exception as e:
            # Silently fall back to CPU for common GPU errors
            pass
    except Exception as e:
        # Silently fall back to CPU if GPU is not available or initialization fails
        pass

    # If we reached here, GPU processing failed - use CPU implementation
    return _process_team_batch_cpu(start_idx, end_idx, skill_matrix, member_matrix)


def _process_team_batch_cpu(start_idx, end_idx, skill_matrix, member_matrix):
    """CPU implementation of batch processing"""
    batch_skill_rows = skill_matrix[start_idx:end_idx]
    batch_member_rows = member_matrix[start_idx:end_idx]

    skills_per_team_batch = []
    experts_per_team_batch = []
    skill_indices_per_team_batch = []
    expert_indices_per_team_batch = []
    zero_skill_teams_batch = 0
    zero_expert_teams_batch = 0
    min_skills_batch = float("inf")
    max_skills_batch = 0
    min_experts_batch = float("inf")
    max_experts_batch = 0
    dup_teams_batch = 0
    seen_configs_batch = {}

    # Process each team in the batch
    for i in range(end_idx - start_idx):
        skill_row = batch_skill_rows[i]
        member_row = batch_member_rows[i]

        # Get skill and expert indices
        skill_indices = tuple(skill_row.nonzero()[1])
        expert_indices = tuple(member_row.nonzero()[1])

        # Store skill and expert indices
        skill_indices_per_team_batch.append(skill_indices)
        expert_indices_per_team_batch.append(expert_indices)

        # Count skills and experts
        num_skills = len(skill_indices)
        num_experts = len(expert_indices)

        skills_per_team_batch.append(num_skills)
        experts_per_team_batch.append(num_experts)

        # Update min/max values
        if num_skills == 0:
            zero_skill_teams_batch += 1
        else:
            min_skills_batch = min(min_skills_batch, num_skills)
            max_skills_batch = max(max_skills_batch, num_skills)

        if num_experts == 0:
            zero_expert_teams_batch += 1
        else:
            min_experts_batch = min(min_experts_batch, num_experts)
            max_experts_batch = max(max_experts_batch, num_experts)

        # Track team configuration for duplicate detection
        team_config = (skill_indices, expert_indices)
        team_idx = i + start_idx

        if team_config in seen_configs_batch:
            dup_teams_batch += 1
        else:
            seen_configs_batch[team_config] = team_idx

    # Handle case where all teams have zero skills/experts
    if min_skills_batch == float("inf"):
        min_skills_batch = 0
    if min_experts_batch == float("inf"):
        min_experts_batch = 0

    # Sets to track unique skill and expert IDs used in this batch
    unique_skills_batch = set()
    unique_experts_batch = set()

    for skill_indices in skill_indices_per_team_batch:
        unique_skills_batch.update(skill_indices)

    for expert_indices in expert_indices_per_team_batch:
        unique_experts_batch.update(expert_indices)

    return (
        skills_per_team_batch,
        experts_per_team_batch,
        skill_indices_per_team_batch,
        expert_indices_per_team_batch,
        zero_skill_teams_batch,
        zero_expert_teams_batch,
        max_skills_batch,
        min_skills_batch,
        max_experts_batch,
        min_experts_batch,
        dup_teams_batch,
        seen_configs_batch,
        unique_skills_batch,
        unique_experts_batch,
    )


def parse_gpu_mode(mode_str):
    """
    Parse the GPU mode string to extract the mode and device indices

    Args:
        mode_str: Mode string in format "cpu", "gpu", "gpu=N", "gpu=all", etc.

    Returns:
        Tuple of (mode, device_indices)
    """
    if not mode_str or mode_str.lower() == "cpu":
        return "cpu", None

    # Check if this is a GPU mode
    if mode_str.lower().startswith("gpu"):
        # Check if CUDA is available before proceeding
        try:
            import cupy as cp

            # Try to get device count to verify CUDA is working
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                tprint("Warning: No CUDA devices found. Falling back to CPU mode.")
                return "cpu", None
        except Exception as e:
            tprint(
                f"Warning: CUDA initialization failed: {str(e)}. Falling back to CPU mode."
            )
            return "cpu", None

        # Check if specific devices are specified
        if "=" in mode_str:
            device_part = mode_str.split("=", 1)[1].strip()

            # Use the parse_gpu_string function for consistent handling
            devices = parse_gpu_string(device_part)

            # Validate devices if they're specified as indices
            if isinstance(devices, list):
                try:
                    import cupy as cp

                    device_count = cp.cuda.runtime.getDeviceCount()
                    valid_devices = [d for d in devices if d < device_count]

                    if not valid_devices:
                        tprint(
                            f"Warning: No valid GPU devices among {devices}. Falling back to first GPU."
                        )
                        return "gpu", "first"

                    if len(valid_devices) < len(devices):
                        invalid = [d for d in devices if d >= device_count]
                        tprint(
                            f"Warning: Invalid GPU devices {invalid} (max index: {device_count-1})"
                        )
                        tprint(f"Using only valid devices: {valid_devices}")
                        devices = valid_devices
                except Exception as e:
                    # If we can't validate, just pass through the devices and let later code handle errors
                    tprint(f"Warning: Couldn't validate GPU devices: {str(e)}")

            return "gpu", devices
        else:
            # No specific device, use first available
            return "gpu", "first"

    # Default to CPU mode for unknown strings
    tprint(f"Unknown mode string '{mode_str}'. Falling back to CPU mode.")
    return "cpu", None


def generate_distribution_charts(stats, out_dir, dataset_name, n_jobs=1, use_gpu=False):
    """
    Generate distribution charts for skills and experts based on stats.
    Creates more advanced visualizations matching utils/data-reports.py style.

    Args:
        stats (dict): Dictionary containing team statistics.
        out_dir (str or Path): Output directory to save charts.
        dataset_name (str): Name of the dataset, used in file names.
        n_jobs (int): Number of parallel jobs for data preparation.
        use_gpu (bool): Whether GPU acceleration is available.

    Returns:
        dict: A dictionary with paths to generated chart images with keys:
              'skills_histogram', 'skills_compact', 'skills_heatmap',
              'experts_histogram', 'experts_compact', 'experts_heatmap'.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    from collections import Counter
    import time
    from pathlib import Path

    # Ensure output directory is a Path object
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    charts = {}
    chart_start_time = time.time()

    # Get data from stats
    skills_per_team = stats.get("skills_per_team", [])
    experts_per_team = stats.get("experts_per_team", [])
    all_skill_indices = stats.get("all_skill_indices", [])
    all_expert_indices = stats.get("all_expert_indices", [])

    # Create a progress bar for the chart generation process
    chart_types = [
        "skills_histogram",
        "skills_compact",
        "skills_heatmap",
        "experts_histogram",
        "experts_compact",
        "experts_heatmap",
    ]

    with tqdm(
        total=len(chart_types), desc="Generating charts", unit="chart", ncols=100
    ) as chart_progress:
        # Calculate teams per skill (inverse of skills per team)
        teams_per_skill = []
        for skill_idx in range(stats.get("unique_skills", 0)):
            # Count how many teams have this skill
            count = sum(1 for indices in all_skill_indices if skill_idx in indices)
            teams_per_skill.append(count)

        # Calculate teams per expert (inverse of experts per team)
        expert_participation = stats.get("expert_participation", {})
        teams_per_expert = list(expert_participation.values())

        # Plot title - use lowercase for consistency
        plot_title = dataset_name.lower()

        # 1. SKILLS CHARTS

        # 1a. Skills Histogram
        tprint("Generating skills histogram...")
        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(skills_per_team, bins=20, color="blue", alpha=0.7, edgecolor="black")
        plt.title(
            f"Distribution of Number of Skills per Team ({dataset_name.upper()})",
            fontsize=14,
        )
        plt.xlabel("Number of Skills", fontsize=12)
        plt.ylabel("Number of Teams", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Calculate coefficient of variation to determine distribution type
        skills_mean = np.mean(skills_per_team) if skills_per_team else 0
        skills_std = np.std(skills_per_team) if skills_per_team else 0
        skills_cv = skills_std / skills_mean if skills_mean > 0 else 0
        distribution_type = "Long-tailed" if skills_cv > 1 else "Relatively Uniform"

        # Add annotation about distribution type
        plt.annotate(
            f"Distribution type: {distribution_type} (CV={skills_cv:.2f})",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        # Add GPU info if using GPU
        if use_gpu and CUPY_AVAILABLE:
            plt.figtext(
                0.02, 0.02, "Generated with GPU acceleration", fontsize=8, color="gray"
            )

        # Save the figure
        skills_hist_path = out_dir / f"{dataset_name}_skills_histogram.png"
        plt.tight_layout()
        plt.savefig(skills_hist_path)
        plt.close()
        charts["skills_histogram"] = str(skills_hist_path)
        chart_progress.update(1)

        # 1b. Skills Compact Plot (log-log scale scatter plot)
        tprint("Generating skills compact plot...")
        plt.figure(figsize=(5, 5))  # Square figure

        # Count occurrence of each team count
        skill_team_count_distribution = Counter(teams_per_skill)

        # Sort by number of teams
        x_values = sorted(skill_team_count_distribution.keys())
        y_values = [skill_team_count_distribution[x] for x in x_values]

        # Plot with 'x' markers
        plt.scatter(x_values, y_values, marker="x", color="blue", alpha=1.0, s=50)

        # Add dataset name to top right
        plt.text(
            0.85,
            0.9,
            plot_title,
            transform=plt.gca().transAxes,
            fontsize=18,
            ha="center",
            va="center",
        )

        # Set log scales for both axes
        plt.xscale("log")
        plt.yscale("log")

        # Labels
        plt.xlabel("#teams", fontsize=14, fontweight="bold")
        plt.ylabel("#skills", fontsize=14, fontweight="bold")

        # Grid
        plt.grid(True, linestyle="-", alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save the figure
        skills_compact_path = out_dir / f"{dataset_name}_skills_compact.png"
        plt.savefig(skills_compact_path)
        plt.close()
        charts["skills_compact"] = str(skills_compact_path)
        chart_progress.update(1)

        # 1c. Skills Heatmap (hexbin)
        tprint("Generating skills heatmap...")

        # Create data for hexbin
        max_skills_to_show = min(stats.get("unique_skills", 100), 100)
        max_teams_to_show = min(len(skills_per_team), 100)

        # Generate (x,y) points for the hexbin
        x_points = []  # Team indices
        y_points = []  # Skill indices

        # For each skill, find which teams have it
        for skill_idx in range(max_skills_to_show):
            for team_idx in range(max_teams_to_show):
                if (
                    team_idx < len(all_skill_indices)
                    and skill_idx in all_skill_indices[team_idx]
                ):
                    x_points.append(team_idx)
                    y_points.append(skill_idx)

        # Create hexbin plot
        plt.figure(figsize=(12, 8))

        if x_points and y_points:  # Only create hexbin if we have data points
            hb = plt.hexbin(x_points, y_points, gridsize=20, cmap="viridis", mincnt=1)
            plt.colorbar(hb, label="Concentration")

        plt.title(
            f"Skills-Teams Relationship Heatmap ({dataset_name.upper()})", fontsize=16
        )
        plt.xlabel("Team Index", fontsize=14)
        plt.ylabel("Skill Index", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        skills_heatmap_path = out_dir / f"{dataset_name}_skills_heatmap.png"
        plt.savefig(skills_heatmap_path)
        plt.close()
        charts["skills_heatmap"] = str(skills_heatmap_path)
        chart_progress.update(1)

        # 2. EXPERTS CHARTS

        # 2a. Experts Histogram
        tprint("Generating experts histogram...")
        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(
            experts_per_team, bins=20, color="firebrick", alpha=0.7, edgecolor="black"
        )
        plt.title(
            f"Distribution of Number of Experts per Team ({dataset_name.upper()})",
            fontsize=14,
        )
        plt.xlabel("Number of Experts", fontsize=12)
        plt.ylabel("Number of Teams", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Calculate coefficient of variation
        experts_mean = np.mean(experts_per_team) if experts_per_team else 0
        experts_std = np.std(experts_per_team) if experts_per_team else 0
        experts_cv = experts_std / experts_mean if experts_mean > 0 else 0
        distribution_type = "Long-tailed" if experts_cv > 1 else "Relatively Uniform"

        # Add annotation about distribution type
        plt.annotate(
            f"Distribution type: {distribution_type} (CV={experts_cv:.2f})",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        # Add GPU info if using GPU
        if use_gpu and CUPY_AVAILABLE:
            plt.figtext(
                0.02, 0.02, "Generated with GPU acceleration", fontsize=8, color="gray"
            )

        # Save the figure
        experts_hist_path = out_dir / f"{dataset_name}_experts_histogram.png"
        plt.tight_layout()
        plt.savefig(experts_hist_path)
        plt.close()
        charts["experts_histogram"] = str(experts_hist_path)
        chart_progress.update(1)

        # 2b. Experts Compact Plot
        tprint("Generating experts compact plot...")
        plt.figure(figsize=(5, 5))  # Square figure

        # Count occurrence of each team count
        expert_team_count_distribution = Counter(teams_per_expert)

        # Sort by number of teams
        x_values = sorted(expert_team_count_distribution.keys())
        y_values = [expert_team_count_distribution[x] for x in x_values]

        # Plot with 'x' markers
        plt.scatter(x_values, y_values, marker="x", color="firebrick", alpha=1.0, s=50)

        # Add dataset name to top right
        plt.text(
            0.85,
            0.9,
            plot_title,
            transform=plt.gca().transAxes,
            fontsize=18,
            ha="center",
            va="center",
        )

        # Set log scales for both axes
        plt.xscale("log")
        plt.yscale("log")

        # Labels
        plt.xlabel("#teams", fontsize=14, fontweight="bold")
        plt.ylabel("#experts", fontsize=14, fontweight="bold")

        # Grid
        plt.grid(True, linestyle="-", alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save the figure
        experts_compact_path = out_dir / f"{dataset_name}_experts_compact.png"
        plt.savefig(experts_compact_path)
        plt.close()
        charts["experts_compact"] = str(experts_compact_path)
        chart_progress.update(1)

        # 2c. Experts Heatmap
        tprint("Generating experts heatmap...")

        # Sort experts by activity (teams per expert)
        expert_activity = [
            (idx, expert_participation.get(idx, 0))
            for idx in range(stats.get("unique_experts", 0))
        ]
        expert_activity.sort(key=lambda x: x[1], reverse=True)
        most_active_experts = [idx for idx, _ in expert_activity]

        # Create data for hexbin
        max_experts_to_show = min(len(most_active_experts), 100)
        max_teams_to_show = min(len(experts_per_team), 100)

        # Generate (x,y) points for the hexbin
        x_points = []  # Team indices
        y_points = []  # Expert indices (by activity rank)

        # For each expert among the most active, find which teams they're in
        for i, expert_idx in enumerate(most_active_experts[:max_experts_to_show]):
            for team_idx in range(max_teams_to_show):
                if (
                    team_idx < len(all_expert_indices)
                    and expert_idx in all_expert_indices[team_idx]
                ):
                    x_points.append(team_idx)
                    y_points.append(i)  # Use rank position for better visualization

        # Create hexbin plot
        plt.figure(figsize=(12, 8))

        if x_points and y_points:  # Only create hexbin if we have data points
            hb = plt.hexbin(x_points, y_points, gridsize=20, cmap="plasma", mincnt=1)
            plt.colorbar(hb, label="Concentration")

        plt.title(
            f"Experts-Teams Relationship Heatmap ({dataset_name.upper()})", fontsize=16
        )
        plt.xlabel("Team Index", fontsize=14)
        plt.ylabel("Expert Index (by activity rank)", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        experts_heatmap_path = out_dir / f"{dataset_name}_experts_heatmap.png"
        plt.savefig(experts_heatmap_path)
        plt.close()
        charts["experts_heatmap"] = str(experts_heatmap_path)
        chart_progress.update(1)

    # Report total chart generation time
    total_chart_time = time.time() - chart_start_time
    tprint(f"All chart generation completed in {total_chart_time:.2f} seconds")

    return charts


def analyze_duplicate_detection(skill_matrix, member_matrix):
    """
    Analyze duplicate detection to match data-filter.py approach

    Args:
        skill_matrix: Sparse matrix of skills
        member_matrix: Sparse matrix of members

    Returns:
        int: Number of duplicates found
    """
    tprint("Analyzing duplicate detection using data-filter.py approach...")

    num_teams = skill_matrix.shape[0]
    seen_configs = set()
    duplicate_count = 0

    # Add a progress bar for the duplicate detection process
    with tqdm(
        total=num_teams, desc="Checking for duplicates", unit="teams", ncols=100
    ) as dup_progress:
        for i in range(num_teams):
            if i % 100000 == 0 and i > 0:
                tprint(f"  Processed {i}/{num_teams} teams ({i/num_teams*100:.1f}%)")

            skill_row = skill_matrix[i]
            member_row = member_matrix[i]

            skill_indices = tuple(skill_row.nonzero()[1])
            member_indices = tuple(member_row.nonzero()[1])

            team_config = (skill_indices, member_indices)

            if team_config in seen_configs:
                duplicate_count += 1
            else:
                seen_configs.add(team_config)

            # Update progress bar every 1000 teams to avoid slowdown
            if i % 1000 == 0:
                dup_progress.update(1000)

        # Update remaining teams
        dup_progress.update(num_teams % 1000)

    tprint(f"Found {duplicate_count} duplicates using data-filter.py approach")
    return duplicate_count


def generate_markdown_report(stats, report_params, out_file=None):
    """
    Generate a markdown report file with detailed statistics.

    Args:
        stats (dict): Statistics dictionary generated by analyze_teams
        report_params (dict): Dictionary with additional parameters for the report:
            - dataset_name: Name of the dataset
            - filter_info: Dictionary of filtering information
            - removed_items: List of removed items during filtering
        out_file (str, optional): Path to output file. If None, a default path is used.

    Returns:
        str: Path to the generated markdown file
    """
    # Extract parameters
    dataset_name = report_params.get("dataset_name", "unknown")
    filter_info = report_params.get("filter_info", {})
    removed_items = report_params.get("removed_items", [])

    # If no output file is specified, create a default one
    if out_file is None:
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True, parents=True)
        out_file = reports_dir / f"{dataset_name.lower()}_statistics.md"
    else:
        out_file = Path(out_file)

    # Get the current datetime
    now = datetime.now()
    if PYTZ_AVAILABLE:
        try:
            eastern = pytz.timezone("US/Eastern")
            now = now.astimezone(eastern)
        except Exception:
            # If timezone conversion fails, use UTC
            pass

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Prepare team statistics
    n_teams = stats["n_teams"]
    zero_skill_teams = stats["zero_skill_teams"]
    zero_expert_teams = stats["zero_expert_teams"]
    dup_teams = stats["dup_teams"]
    unique_team_configs = stats["n_unique_team_configs"]

    # Calculate percentages
    zero_skill_percent = (zero_skill_teams / n_teams) * 100 if n_teams > 0 else 0
    zero_expert_percent = (zero_expert_teams / n_teams) * 100 if n_teams > 0 else 0
    dup_teams_percent = (dup_teams / n_teams) * 100 if n_teams > 0 else 0
    unique_teams_percent = (unique_team_configs / n_teams) * 100 if n_teams > 0 else 0

    # Skill statistics
    min_skills = stats["min_skills"]
    max_skills = stats["max_skills"]
    min_skill_teams = sum(1 for x in stats["skills_per_team"] if x == min_skills)
    max_skill_teams = sum(1 for x in stats["skills_per_team"] if x == max_skills)
    min_skill_percent = (min_skill_teams / n_teams) * 100 if n_teams > 0 else 0
    max_skill_percent = (max_skill_teams / n_teams) * 100 if n_teams > 0 else 0

    # Expert statistics
    min_experts = stats["min_experts"]
    max_experts = stats["max_experts"]
    min_expert_teams = sum(1 for x in stats["experts_per_team"] if x == min_experts)
    max_expert_teams = sum(1 for x in stats["experts_per_team"] if x == max_experts)
    min_expert_percent = (min_expert_teams / n_teams) * 100 if n_teams > 0 else 0
    max_expert_percent = (max_expert_teams / n_teams) * 100 if n_teams > 0 else 0

    # Expert participation statistics
    min_participation = stats["min_participation"]
    max_participation = stats["max_participation"]
    min_participation_count = stats["min_participation_count"]
    max_participation_count = stats["max_participation_count"]

    # Calculate skill and expert distributions for histograms
    skills_per_team = stats["skills_per_team"]
    experts_per_team = stats["experts_per_team"]

    # Get the mean and standard deviation for skills and experts
    skills_mean = np.mean(skills_per_team) if skills_per_team else 0
    skills_std = np.std(skills_per_team) if skills_per_team else 0
    experts_mean = np.mean(experts_per_team) if experts_per_team else 0
    experts_std = np.std(experts_per_team) if experts_per_team else 0

    # Generate the markdown content
    content = [
        f"# {dataset_name.upper()} Dataset Statistics",
        f"Generated on: {timestamp}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Teams | {n_teams:,} |",
        f"| Total Skills | {stats['unique_skills']:,} |",
        f"| Total Experts | {stats['unique_experts']:,} |",
        "",
        "## Team Composition",
        "",
        "### Teams Breakdown",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Unique Teams | {unique_team_configs:,} ({unique_teams_percent:.1f}%) |",
        f"| Duplicate Teams | {dup_teams:,} ({dup_teams_percent:.1f}%) |",
        f"| Teams with No Skills | {zero_skill_teams:,} ({zero_skill_percent:.1f}%) |",
        f"| Teams with No Experts | {zero_expert_teams:,} ({zero_expert_percent:.1f}%) |",
        "",
        "### Skills per Team",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Minimum Skills | {min_skills:,} ({min_skill_teams:,} teams, {min_skill_percent:.1f}%) |",
        f"| Maximum Skills | {max_skills:,} ({max_skill_teams:,} teams, {max_skill_percent:.1f}%) |",
        f"| Average Skills | {skills_mean:.4f} |",
        f"| Standard Deviation | {skills_std:.4f} |",
        "",
        "### Experts per Team",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Minimum Experts | {min_experts:,} ({min_expert_teams:,} teams, {min_expert_percent:.1f}%) |",
        f"| Maximum Experts | {max_experts:,} ({max_expert_teams:,} teams, {max_expert_percent:.1f}%) |",
        f"| Average Experts | {experts_mean:.4f} |",
        f"| Standard Deviation | {experts_std:.4f} |",
        "",
        "### Expert Participation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Minimum Teams per Expert | {min_participation:,} ({min_participation_count:,} experts) |",
        f"| Maximum Teams per Expert | {max_participation:,} ({max_participation_count:,} experts) |",
    ]

    # Calculate average skills per expert and average teams per expert
    # Average teams per expert
    expert_participation_values = list(stats["expert_participation"].values())
    avg_teams_per_expert = (
        np.mean(expert_participation_values) if expert_participation_values else 0
    )

    # Average skills per expert - calculate based on team information
    # This is more complex as we need to map which skills each expert has across teams
    expert_to_skills = {}

    # Process all teams to gather expert-skill associations
    for i in range(len(stats["all_expert_indices"])):
        team_experts = stats["all_expert_indices"][i]
        team_skills = stats["all_skill_indices"][i]

        # For each expert in this team, add all team skills to their skill set
        for expert_idx in team_experts:
            if expert_idx not in expert_to_skills:
                expert_to_skills[expert_idx] = set()
            expert_to_skills[expert_idx].update(team_skills)

    # Calculate average number of skills per expert
    skills_per_expert = [len(skills) for skills in expert_to_skills.values()]
    avg_skills_per_expert = np.mean(skills_per_expert) if skills_per_expert else 0

    # Add the additional rows to Expert Participation table
    content.extend(
        [
            f"| Average Skills per Expert | {avg_skills_per_expert:.4f} |",
            f"| Average Teams per Expert | {avg_teams_per_expert:.4f} |",
            "",
        ]
    )

    # Add filtering information if available
    if filter_info:
        content.extend(
            [
                "## Filtering Information",
                "",
                f"| Filter | Value |",
                f"|--------|-------|",
            ]
        )

        for key, value in filter_info.items():
            content.append(f"| {key} | {value} |")

        content.append("")

    # Add list of removed items if available
    if removed_items:
        content.extend(
            [
                "### Removed Items",
                "",
            ]
        )

        for item in removed_items:
            content.append(f"- {item}")

        content.append("")

    # Add visualization reference
    content.extend(
        [
            "## Visualizations",
            "",
            "The following visualizations are available in the reports directory:",
            "",
            "### Skills Distribution",
            "",
            f"- `{dataset_name}_skills_histogram.png`: Histogram of skills per team",
            f"- `{dataset_name}_skills_compact.png`: Log-log plot of skills distribution",
            f"- `{dataset_name}_skills_heatmap.png`: Heatmap of skill-team relationships",
            "",
            "### Experts Distribution",
            "",
            f"- `{dataset_name}_experts_histogram.png`: Histogram of experts per team",
            f"- `{dataset_name}_experts_compact.png`: Log-log plot of experts distribution",
            f"- `{dataset_name}_experts_heatmap.png`: Heatmap of expert-team relationships",
            "",
            "## Notes on Statistics",
            "",
            "- **Average Skills per Team**: This is calculated as the mean number of skills across all teams. Each team contributes one count to this average, regardless of team size.",
            "- **Average Experts per Team**: Similarly, this is the mean number of experts across all teams.",
            "- **Total Skills/Experts vs. Averages**: The total number of unique skills/experts represents distinct entities in the dataset, while averages show how many are typically associated with each team.",
            "- **Standard Deviation**: Measures the variation in the distribution. Higher values indicate more diversity in team compositions.",
        ]
    )

    # Write the markdown file
    with open(out_file, "w") as f:
        f.write("\n".join(content))

    tprint(f"Markdown report generated at {out_file}")
    return str(out_file)


def main():
    # Start the timer
    start_time = time.time()

    tprint("Starting data reports process...")

    parser = argparse.ArgumentParser(
        description="Generate reports and visualizations for teams data."
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Path to the input file or directory",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=False,
        help="Dataset name (gith, dblp, etc.)",
    )

    # Optional arguments
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        help="Output directory for reports (default: derived from input path)",
    )

    parser.add_argument(
        "-mode",
        "--mode",
        type=str,
        default="gpu",
        help="Processing mode: gpu (default - first GPU), cpu, gpu=all (all GPUs), gpu=N (specific GPU), gpu=N,M (multiple GPUs)",
    )

    # Update default threads based on mode
    global DEFAULT_THREADS
    DEFAULT_THREADS = get_default_threads("gpu")  # Default to GPU mode initially

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of parallel threads to use (default: auto-detected based on mode)",
    )

    parser.add_argument(
        "-fs",
        "--filter-stats",
        type=str,
        default="filter_stats.txt",
        help="Path to filter stats file (default: filter_stats.txt in the same directory as the input file)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Parse the mode to extract GPU device indices if specified
    mode, device_indices = parse_gpu_mode(args.mode)

    # If GPU devices were specified, set them globally
    if mode == "gpu":
        global SELECTED_GPU_DEVICES
        SELECTED_GPU_DEVICES = device_indices

    # Update default threads based on selected mode if user didn't specify
    if args.threads == DEFAULT_THREADS:  # Only if user didn't override
        DEFAULT_THREADS = get_nthreads()  # Use the enhanced thread_utils function
        args.threads = DEFAULT_THREADS
        if mode == "cpu":
            tprint(
                f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {args.threads} threads from thread_utils"
            )
        else:
            tprint(f"Using {args.threads} CPU threads for GPU-accelerated mode")
    else:
        tprint(f"Using user-specified {args.threads} threads")

    # Set debug mode if requested
    if args.debug:
        global DEBUG
        DEBUG = True

    input_path = Path(args.input_path)

    # If input is a directory, look for teamsvecs.pkl inside it
    if input_path.is_dir():
        teamsvecs_path = input_path / "teamsvecs.pkl"
        if not teamsvecs_path.exists():
            raise FileNotFoundError(f"Could not find teamsvecs.pkl in {input_path}")
        input_path = teamsvecs_path

    # Derive output directory from input path if not specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Just use the parent directory of the input file
        output_dir = input_path.parent

    # If dataset name is not provided, try to infer from the path
    if not args.dataset:
        # Try to infer dataset name from path
        path_str = str(input_path).lower()
        if "gith" in path_str:
            dataset_name = "gith"
        elif "dblp" in path_str:
            dataset_name = "dblp"
        elif "uspt" in path_str:
            dataset_name = "uspt"
        elif "imdb" in path_str:
            dataset_name = "imdb"
        elif "cf9" in path_str:
            dataset_name = "cf9"
        else:
            dataset_name = "unknown"

        tprint(f"Inferred dataset name: {dataset_name}")
    else:
        dataset_name = args.dataset.lower()

    # Load the data
    tprint(f"Loading data from {input_path}...")
    with tqdm(total=1, desc="Loading data", ncols=100) as pbar:
        try:
            with open(input_path, "rb") as f:
                teamsvecs = pickle.load(f)
                teamsvecs_data = teamsvecs
                pbar.update(1)
        except Exception as e:
            tprint(f"Error loading teamsvecs data: {e}")
            sys.exit(1)

    # Check for filter stats file - use the specified file or default
    filter_stats_path = None

    # If a relative path was provided, look relative to the input file's directory
    if not os.path.isabs(args.filter_stats):
        filter_stats_path = input_path.parent / args.filter_stats
    else:
        # If an absolute path was provided, use it directly
        filter_stats_path = Path(args.filter_stats)

    filter_info = {}
    removed_items = []

    if filter_stats_path.exists():
        tprint(f"Found filter stats at {filter_stats_path}")
        try:
            with open(filter_stats_path, "r") as f:
                lines = f.readlines()
                # Flag to track when we're in the "Removed:" section
                in_removed_section = False

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines

                    # Check if we're entering the "Removed:" section
                    if line == "Removed:":
                        in_removed_section = True
                        continue

                    # If we're in the "Removed:" section and the line starts with "- "
                    if in_removed_section and line.startswith("- "):
                        removed_item = line[2:]  # Remove the "- " prefix
                        removed_items.append(removed_item)
                    # If we're in the removed section but the line doesn't start with "- "
                    # and it's not empty, we've exited the removed section
                    elif in_removed_section and not line.startswith("- "):
                        in_removed_section = False

                    # Extract other key-value pairs for debugging
                    if ":" in line and not in_removed_section:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                filter_info[key] = value

        except Exception as e:
            tprint(f"Warning: Could not parse filter stats file: {e}")

    # Add debug flag to compare duplicate detection with filter.py
    if args.debug and "Duplicate teams" in filter_info:
        tprint(
            f"\nDEBUG: Filter stats reported {filter_info['Duplicate teams']} duplicates"
        )
        tprint(
            "DEBUG: Will compare with generate_reports.py duplicate detection method"
        )

    tprint("Analyzing teams data...")
    stats = analyze_teams(teamsvecs, n_jobs=args.threads)

    # Add diagnostic duplicate detection when in debug mode
    if args.debug:
        tprint("\nRunning diagnostic duplicate detection...")
        filter_style_dup_count = analyze_duplicate_detection(
            teamsvecs["skill"], teamsvecs["member"]
        )
        tprint(f"Filter-style duplicate count: {filter_style_dup_count}")
        tprint(f"Reports-style duplicate count: {stats['dup_teams']}")

    # Compare duplicate counts if debugging
    if args.debug and "Duplicate teams" in filter_info:
        filter_dup_count = int(filter_info["Duplicate teams"].replace(",", ""))
        reports_dup_count = stats["dup_teams"]
        tprint(f"\nDEBUG: Duplicate team comparison:")
        tprint(f"  - data-filter.py found: {filter_dup_count} duplicates")
        tprint(f"  - generate_reports.py found: {reports_dup_count} duplicates")
        tprint(f"  - Difference: {abs(filter_dup_count - reports_dup_count)} teams")
        if filter_dup_count != reports_dup_count:
            tprint(f"  - Possible reasons for difference:")
            tprint(
                f"    1. Different definition of duplicates (filter looks at both experts & skills, reports might have additional criteria)"
            )
            tprint(
                f"    2. Different ordering/processing of teams during duplicate detection"
            )
            tprint(
                f"    3. Filter removes duplicates before analysis, reports counts after they're already removed"
            )

    # Create output file in the same directory as the input file
    input_filename = input_path.name
    output_suffix = (
        f'_{input_filename.replace(".pkl", "")}'
        if input_filename != "teamsvecs.pkl"
        else ""
    )

    # Create reports directory next to the input file
    stats_reports_dir = input_path.parent / "reports"
    stats_reports_dir.mkdir(exist_ok=True, parents=True)

    # Generate CSV report
    output_file = stats_reports_dir / f"{dataset_name}_team_stats{output_suffix}.csv"

    tprint(f"Writing report to {output_file}...")
    sys.stdout.flush()
    with tqdm(total=1, desc="Writing report", ncols=100) as pbar:
        with open(output_file, "w") as f:
            # Basic stats with dataset name
            f.write(",dataset,#teams,#skills,#experts\n")
            f.write(
                f',{dataset_name.upper()},{stats["n_teams"]},{stats["unique_skills"]},{stats["max_experts"]}\n'
            )
            f.write(",,,,\n")

            # Zero teams and duplicates stats
            f.write(
                ",#zero_skill_teams,#zero_expert_teams,#dup_teams,#unique_team_configs\n"
            )
            zero_skill_percent = (stats["zero_skill_teams"] / stats["n_teams"]) * 100
            zero_expert_percent = (stats["zero_expert_teams"] / stats["n_teams"]) * 100
            dup_teams_percent = (stats["dup_teams"] / stats["n_teams"]) * 100
            unique_teams = stats["n_unique_team_configs"]
            unique_teams_percent = (unique_teams / stats["n_teams"]) * 100

            f.write(
                f',{stats["zero_skill_teams"]} ({zero_skill_percent:.1f}% of teams),'
                + f'{stats["zero_expert_teams"]} ({zero_expert_percent:.1f}% of teams),'
                + f'{stats["dup_teams"]} ({dup_teams_percent:.1f}% of teams),'
                + f"{unique_teams} ({unique_teams_percent:.1f}% of teams)\n"
            )
            f.write(",,,,\n")

            # Team size stats
            f.write(
                ",#min_team_experts,#max_team_experts,#min_team_skill,#max_team_skill\n"
            )

            # Count teams with min/max experts/skills
            min_expert_teams = sum(
                1 for x in stats["experts_per_team"] if x == stats["min_experts"]
            )
            max_expert_teams = sum(
                1 for x in stats["experts_per_team"] if x == stats["max_experts"]
            )
            min_skill_teams = sum(
                1 for x in stats["skills_per_team"] if x == stats["min_skills"]
            )
            max_skill_teams = sum(
                1 for x in stats["skills_per_team"] if x == stats["max_skills"]
            )

            min_expert_percent = (min_expert_teams / stats["n_teams"]) * 100
            max_expert_percent = (max_expert_teams / stats["n_teams"]) * 100
            min_skill_percent = (min_skill_teams / stats["n_teams"]) * 100
            max_skill_percent = (max_skill_teams / stats["n_teams"]) * 100

            f.write(
                f',{stats["min_experts"]} ({min_expert_teams} teams~{min_expert_percent:.1f}%),'
                + f'{stats["max_experts"]} ({max_expert_teams} teams~{max_expert_percent:.1f}%),'
                + f'{stats["min_skills"]} ({min_skill_teams} teams~{min_skill_percent:.1f}%),'
                + f'{stats["max_skills"]} ({max_skill_teams} teams~{max_skill_percent:.1f}%)\n'
            )

            # Expert participation stats - fix misaligned column headings
            f.write(
                ",min_experts_per_team,max_experts_per_team,min_skills_per_team,max_skills_per_team\n"
            )

            # Count experts with min/max team participation for the next section
            # Handle missing lists - use empty lists if not available
            min_team_experts_list = stats.get("min_team_experts_list", [])
            max_team_experts_list = stats.get("max_team_experts_list", [])
            min_team_experts = len(min_team_experts_list)
            max_team_experts = len(max_team_experts_list)
            min_exp_percent = (
                (min_team_experts / stats["unique_experts"]) * 100
                if stats["unique_experts"] > 0
                else 0
            )
            max_exp_percent = (
                (max_team_experts / stats["unique_experts"]) * 100
                if stats["unique_experts"] > 0
                else 0
            )

            f.write(
                f'{stats["min_experts"]} ({min_team_experts} experts ~ {min_exp_percent:.1f}%),'
                + f'{stats["max_experts"]} ({max_team_experts} experts ~ {max_exp_percent:.1f}%),'
                + f'{stats["min_skills"]} ({min_skill_teams} teams~{min_skill_percent:.1f}%),'
                + f'{stats["max_skills"]} ({max_skill_teams} teams~{max_skill_percent:.1f}%)\n'
            )

            # For detailed section with expert participation stats
            f.write(f"\n### Expert Participation\n\n")
            f.write(f"min_teams_per_expert,max_teams_per_expert\n")
            f.write(
                f'{stats["min_participation"]} ({min_team_experts} experts ~ {min_exp_percent:.1f}%),'
                + f'{stats["max_participation"]} ({max_team_experts} experts ~ {max_exp_percent:.1f}%)\n\n'
            )

            # Write header for detailed stats
            f.write("dup_index,#skills,#experts,skills,experts\n")

            # Write detailed stats for each team
            tprint("Writing detailed team statistics...")
            sys.stdout.flush()
            with tqdm(
                total=len(stats["skills_per_team"]),
                desc="Writing team details",
                mininterval=0.1,
                unit="teams",
                ncols=100,
            ) as team_pbar:
                for i in range(len(stats["skills_per_team"])):
                    # Add 1 to indices to start from 1 instead of 0
                    skill_indices = [
                        f"s{idx+1}" for idx in stats["all_skill_indices"][i]
                    ]
                    expert_indices = [
                        f"m{idx+1}" for idx in stats["all_expert_indices"][i]
                    ]

                    # Use 0 if dup_indices is not available
                    dup_index = 0
                    if "dup_indices" in stats:
                        dup_index = stats["dup_indices"][i]

                    f.write(
                        f'{dup_index},{stats["skills_per_team"][i]},{len(expert_indices)},'
                        + f'[{"-".join(skill_indices)}],'
                        + f'[{"-".join(expert_indices)}]\n'
                    )

                    # Update progress bar
                    team_pbar.update(1)

                    # For very large datasets, update less frequently to avoid slowdown
                    if i % 1000 == 0:
                        team_pbar.refresh()
        pbar.update(1)

    # Generate distribution charts
    tprint("Generating distribution charts...")
    chart_start_time = time.time()
    use_gpu = mode == "gpu"
    if use_gpu:
        if device_indices == "first":
            tprint("GPU mode enabled - using first available GPU")
        elif device_indices == "all":
            tprint("GPU mode enabled - using all available GPUs")
        elif isinstance(device_indices, list):
            gpu_str = ", ".join(map(str, device_indices))
            tprint(f"GPU mode enabled - using specified GPU(s): {gpu_str}")
        else:
            tprint("GPU mode enabled - will try to use GPU acceleration")

    charts = generate_distribution_charts(
        stats, stats_reports_dir, dataset_name, n_jobs=args.threads, use_gpu=use_gpu
    )

    chart_time = time.time() - chart_start_time
    tprint(f"Chart generation completed in {chart_time:.2f} seconds")

    # Generate markdown report
    tprint("Generating markdown report...")

    # Create a params dictionary with necessary info for the markdown report
    report_params = {
        "dataset_name": dataset_name,
        "filter_info": filter_info,
        "removed_items": removed_items,
    }

    # Call with the new signature, providing the output file path
    md_file_path = os.path.join(
        stats_reports_dir, f"{dataset_name.lower()}_statistics.md"
    )
    md_file_path = generate_markdown_report(stats, report_params, out_file=md_file_path)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Analysis complete message
    tprint(f"Analysis complete! All files saved to {stats_reports_dir}:")
    tprint(f"- CSV report: {output_file}")
    tprint(f"- Markdown report: {md_file_path}")
    tprint(f"- Skills distribution charts:")
    tprint(f"  - Histogram: {charts['skills_histogram']}")
    tprint(f"  - Compact: {charts['skills_compact']}")
    tprint(f"  - Heatmap: {charts['skills_heatmap']}")
    tprint(f"- Experts distribution charts:")
    tprint(f"  - Histogram: {charts['experts_histogram']}")
    tprint(f"  - Compact: {charts['experts_compact']}")
    tprint(f"  - Heatmap: {charts['experts_heatmap']}")

    # Report on GPU usage if applicable
    if use_gpu:
        if CUPY_AVAILABLE:
            tprint("Used GPU acceleration for distribution charts")
            # Import cupy here to get memory stats
            try:
                import cupy as cp

                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem_gb = mem_info[0] / (1024**3)
                total_mem_gb = mem_info[1] / (1024**3)
                tprint(
                    f"GPU memory: {free_mem_gb:.1f} GB free / {total_mem_gb:.1f} GB total"
                )
            except Exception as e:
                tprint(f"Could not get GPU memory info: {e}")
        else:
            tprint(
                "GPU acceleration was requested but not available. Used CPU instead."
            )

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    tprint(
        f"Total execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
