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
    set_gpu_devices,
)

# Import parse_gpus_string directly from utils instead of parse_gpu_string from import_gpu_libs
from utils.parse_gpus_string import parse_gpus_string
from cmn_v3.helper_functions.get_gpu_device import get_gpu_device
from cmn_v3.helper_functions.get_default_threads import get_default_threads
from utils.parse_nthreads import parse_nthreads

# Default number of threads for parallel processing
# We'll reduce this if GPU mode is active to prevent memory contention
DEFAULT_THREADS = parse_nthreads()


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

            # Use the parse_gpus_string function for consistent handling
            devices = parse_gpus_string(device_part)

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


def generate_reports(
    input_file=None,
    output_dir=None,
    domain=None,
    threads=None,
    mode="auto",
    depth=1,
    force=False,
    **kwargs,
):
    """
    Generate reports and visualizations for team data.
    This function is a wrapper for the main() function that provides a programmatic interface.

    Args:
        input_file: Path to the teamsvecs.pkl file or directory containing it
        output_dir: Output directory for reports and visualizations
        domain: Domain name (e.g., 'dblp', 'gith')
        threads: Number of threads for parallel processing
        mode: Processing mode ('auto', 'cpu', 'gpu', 'gpu=0', 'gpu=all', etc.)
        depth: Analysis depth (1-3, higher values produce more detailed reports)
        force: Force overwrite of existing files
        **kwargs: Additional arguments to pass to main function

    Returns:
        Dictionary containing paths to generated reports and visualizations
    """
    # Prepare arguments for main function
    import argparse

    args = argparse.Namespace(
        input_file=input_file,
        output_dir=output_dir,
        domain=domain,
        threads=threads if threads is not None else DEFAULT_THREADS,
        mode=mode,
        depth=depth,
        force=force,
        **kwargs,
    )

    # Call main function
    return main(args)


def main(args=None):
    """
    Main entry point for the script.

    Args:
        args: Optional argparse.Namespace object. If None, args will be parsed from command line.

    Returns:
        Dictionary containing paths to generated reports and visualizations
    """
    start_time = time.time()

    # Parse command line arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(
            description="Generate reports and visualizations for team data.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Required arguments
        parser.add_argument(
            "-i",
            "--input-file",
            dest="input_file",
            required=True,
            help="Path to the teamsvecs.pkl file or directory containing it",
        )

        # Optional arguments
        parser.add_argument(
            "-o",
            "--output-dir",
            dest="output_dir",
            help="Output directory for reports (default: same directory as input file)",
        )
        parser.add_argument(
            "-d",
            "--domain",
            dest="domain",
            help="Domain name (optional, will be inferred from filepath if not provided)",
        )
        parser.add_argument(
            "-t",
            "--threads",
            dest="threads",
            type=int,
            default=DEFAULT_THREADS,
            help=f"Number of threads for parallel processing (default: {DEFAULT_THREADS})",
        )
        parser.add_argument(
            "-mode",
            dest="mode",
            default="auto",
            help="Processing mode: 'cpu', 'gpu', 'gpu=0', 'gpu=0,1', 'gpu=all', or 'auto' (default)",
        )
        parser.add_argument(
            "-depth",
            dest="depth",
            type=int,
            choices=[1, 2, 3],
            default=1,
            help="Analysis depth: 1 (basic), 2 (detailed), 3 (comprehensive)",
        )
        parser.add_argument(
            "-force",
            dest="force",
            action="store_true",
            help="Force overwrite of existing files",
        )

        args = parser.parse_args()

    # Process input path and determine output directory and domain
    input_path = Path(args.input_file)
    if not input_path.exists():
        tprint(f"Error: Input file or directory does not exist: {input_path}")
        return None

    # If input is a directory, look for teamsvecs.pkl file
    if input_path.is_dir():
        teamsvecs_path = input_path / "teamsvecs.pkl"
        if not teamsvecs_path.exists():
            tprint(f"Error: Could not find teamsvecs.pkl in directory: {input_path}")
            return None
    else:
        teamsvecs_path = input_path

    # If domain not provided, try to infer from path
    domain_name = args.domain
    if domain_name is None:
        # Try to extract domain from path (dblp, gith, etc.)
        path_parts = str(teamsvecs_path).lower().split(os.path.sep)
        potential_domains = ["dblp", "gith", "so", "nih"]
        for part in path_parts:
            for domain in potential_domains:
                if domain in part:
                    domain_name = domain
                    break
            if domain_name:
                break

        # If still not found, use directory name
        if domain_name is None:
            domain_name = teamsvecs_path.parent.name.lower()
            tprint(f"Domain not provided, using directory name: {domain_name}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = teamsvecs_path.parent / "reports"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    tprint(f"Output directory: {output_dir}")

    # Check if output files already exist and force flag is not set
    output_file = output_dir / f"{domain_name}_statistics.csv"
    md_file_path = output_dir / f"{domain_name}_statistics.md"

    if not args.force and (output_file.exists() or md_file_path.exists()):
        tprint("Output files already exist. Use -force to overwrite.")
        existing_files = []
        if output_file.exists():
            existing_files.append(str(output_file))
        if md_file_path.exists():
            existing_files.append(str(md_file_path))

        # Return existing files without processing
        return {
            "csv_report": str(output_file) if output_file.exists() else None,
            "markdown_report": str(md_file_path) if md_file_path.exists() else None,
            "charts": {},
            "stats": {},
            "execution_time": 0,
        }

    # Set up GPU/CPU mode
    if args.mode.lower() == "auto":
        # Try to initialize GPU, fall back to CPU if not available
        try:
            gpu_mode, gpu_devices = parse_gpu_mode("gpu")
            if gpu_mode == "cpu":
                tprint("GPU not available, using CPU mode")
            else:
                tprint(f"Using GPU mode with devices: {gpu_devices}")
        except Exception as e:
            tprint(f"Error initializing GPU: {str(e)}")
            gpu_mode = "cpu"
            gpu_devices = None
    else:
        gpu_mode, gpu_devices = parse_gpu_mode(args.mode)

    # Import GPU libraries if needed
    if gpu_mode == "gpu":
        # Set the global SELECTED_GPU_DEVICES before calling import_gpu_libs
        global SELECTED_GPU_DEVICES
        SELECTED_GPU_DEVICES = gpu_devices
        set_gpu_devices(SELECTED_GPU_DEVICES)
        tprint(f"Using GPU acceleration with devices: {SELECTED_GPU_DEVICES}")
        use_gpu = True
    else:
        tprint("Using CPU mode for processing")
        use_gpu = False

    # Load the teamsvecs file
    tprint(f"Loading team data from: {teamsvecs_path}")
    try:
        with open(teamsvecs_path, "rb") as f:
            teamsvecs = pickle.load(f)
        tprint(f"Successfully loaded team data of size: {len(teamsvecs)}")
    except Exception as e:
        tprint(f"Error loading team data: {str(e)}")
        return None

    # Determine number of threads to use
    n_jobs = args.threads
    if use_gpu:
        # Reduce thread count when using GPU to avoid memory contention
        n_jobs = min(n_jobs, get_default_threads("gpu"))
        tprint(f"Adjusted thread count for GPU mode: {n_jobs}")

    # Analyze teams
    tprint(f"Analyzing teams with {n_jobs} threads...")
    stats = analyze_teams(teamsvecs, n_jobs=n_jobs)
    tprint(f"Analysis complete. Found {stats['n_teams']} teams.")

    # Generate charts based on the analysis depth
    charts = {}
    if args.depth >= 1:
        tprint("Generating basic charts...")
        charts = generate_distribution_charts(
            stats, output_dir, domain_name, n_jobs=n_jobs, use_gpu=use_gpu
        )

    # Generate additional reports based on depth
    if args.depth >= 2:
        tprint("Generating detailed team report...")
        # Add code here for more detailed reports if needed

    # Generate CSV report
    tprint(f"Generating CSV report at: {output_file}")
    try:
        # Create DataFrame from stats
        report_data = {
            "Metric": [
                "Total Teams",
                "Unique Team Configurations",
                "Duplicate Teams",
                "Zero Skill Teams",
                "Zero Expert Teams",
                "Min Skills Per Team",
                "Max Skills Per Team",
                "Min Experts Per Team",
                "Max Experts Per Team",
                "Min Teams Per Expert",
                "Max Teams Per Expert",
                "Unique Skills",
                "Unique Experts",
            ],
            "Value": [
                stats["n_teams"],
                stats["n_unique_team_configs"],
                stats["dup_teams"],
                stats["zero_skill_teams"],
                stats["zero_expert_teams"],
                stats["min_skills"],
                stats["max_skills"],
                stats["min_experts"],
                stats["max_experts"],
                stats["min_participation"],
                stats["max_participation"],
                stats["unique_skills"],
                stats["unique_experts"],
            ],
        }

        # Create DataFrame and write to CSV
        df = pd.DataFrame(report_data)
        df.to_csv(output_file, index=False)
        tprint(f"CSV report successfully generated at: {output_file}")
    except Exception as e:
        tprint(f"Error generating CSV report: {str(e)}")
        output_file = None

    # Generate markdown report
    tprint(f"Generating markdown report at: {md_file_path}")
    try:
        report_params = {
            "dataset_name": domain_name,
            "filter_info": {},  # Add filtering info if available
            "removed_items": [],  # Add removed items if available
        }

        md_file_path = generate_markdown_report(stats, report_params, md_file_path)
        tprint(f"Markdown report successfully generated at: {md_file_path}")
    except Exception as e:
        tprint(f"Error generating markdown report: {str(e)}")
        md_file_path = None

    # Calculate total execution time
    total_time = time.time() - start_time
    tprint(f"Total execution time: {total_time:.2f} seconds")

    # Return dictionary with paths to generated reports and visualizations
    results = {
        "csv_report": str(output_file) if output_file else None,
        "markdown_report": md_file_path,
        "charts": charts,
        "stats": stats,
        "execution_time": total_time,
    }

    return results


if __name__ == "__main__":
    main()
