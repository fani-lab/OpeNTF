import pickle
import os
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm  # Import tqdm for progress bars
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
from collections import Counter
import seaborn as sns
from scipy.sparse import csr_matrix
import matplotlib.cm as cm
from scipy import stats
from datetime import datetime
import pytz
import time
import multiprocessing
from joblib import Parallel, delayed

# Conditional imports - only try to import GPU libraries if needed
CUPY_AVAILABLE = False
SELECTED_GPU_DEVICES = None  # Will store indices of GPUs to use (if specified)

def try_import_gpu_libs():
    """
    Try to import GPU acceleration libraries and return availability status
    
    Returns:
        bool: True if GPU libraries are available, False otherwise
    """
    global CUPY_AVAILABLE, SELECTED_GPU_DEVICES
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
        
        # Get GPU device information if available
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            
            # Handle the special modes for GPU selection
            if SELECTED_GPU_DEVICES == 'all':
                # Use all available GPUs
                SELECTED_GPU_DEVICES = list(range(device_count))
                tprint(f"Using all {device_count} available GPU devices")
                
            elif SELECTED_GPU_DEVICES == 'first':
                # Use only the first GPU
                SELECTED_GPU_DEVICES = [0]
                tprint(f"Using the first available GPU (index 0)")
            
            # Use only specific devices if specified
            if isinstance(SELECTED_GPU_DEVICES, list):
                # Validate that all specified devices exist
                invalid_devices = [d for d in SELECTED_GPU_DEVICES if d >= device_count]
                if invalid_devices:
                    tprint(f"Warning: GPU devices {invalid_devices} do not exist (max index: {device_count-1})")
                    # Remove invalid devices
                    SELECTED_GPU_DEVICES = [d for d in SELECTED_GPU_DEVICES if d < device_count]
                    if not SELECTED_GPU_DEVICES:
                        tprint(f"No valid GPU devices specified. Falling back to first available GPU.")
                        SELECTED_GPU_DEVICES = [0] if device_count > 0 else None
                
                # If we have valid devices to use, set them up
                if SELECTED_GPU_DEVICES:
                    # Set visible devices (only affects future CUDA operations)
                    device_list_str = ",".join(map(str, SELECTED_GPU_DEVICES))
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"] = device_list_str
                    tprint(f"Limiting to GPU devices: {device_list_str}")
                    
                    # Get information about selected devices
                    device_info = []
                    for device_idx in SELECTED_GPU_DEVICES:
                        device = cp.cuda.Device(device_idx)
                        props = cp.cuda.runtime.getDeviceProperties(device_idx)
                        device_info.append(f"Device {device_idx}: {props['name'].decode('utf-8')} ({props['totalGlobalMem'] / (1024**3):.1f} GB)")
                    
                    tprint(f"GPU acceleration enabled. Using {len(SELECTED_GPU_DEVICES)} specified GPU device(s):")
                    for info in device_info:
                        tprint(f"  {info}")
                    
                    # Use the first device in the list as the primary
                    cp.cuda.Device(SELECTED_GPU_DEVICES[0]).use()
                    tprint(f"Using GPU {SELECTED_GPU_DEVICES[0]} as primary device")
            else:
                # Use all available devices
                device_info = []
                for i in range(device_count):
                    device = cp.cuda.Device(i)
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    device_info.append(f"Device {i}: {props['name'].decode('utf-8')} ({props['totalGlobalMem'] / (1024**3):.1f} GB)")
                
                tprint(f"GPU acceleration enabled. Found {device_count} CUDA device(s):")
                for info in device_info:
                    tprint(f"  {info}")
        except Exception as e:
            tprint(f"GPU acceleration enabled, but couldn't get device info: {e}")
        
        return True
    except ImportError:
        tprint("Warning: CuPy not available. GPU acceleration disabled.")
        tprint("To enable GPU acceleration, install CuPy with: pip install cupy-cuda11x")
        tprint("(Replace 'cuda11x' with your CUDA version)")
        return False
    except Exception as e:
        tprint(f"Error initializing GPU libraries: {e}")
        tprint("GPU acceleration disabled. Falling back to CPU.")
        return False

# Global configuration
DEBUG = False

def get_default_threads(mode='cpu'):
    """
    Determine the default number of threads based on mode
    
    Args:
        mode: 'cpu' or 'gpu'
        
    Returns:
        int: Recommended number of threads
    """
    if mode == 'cpu':
        # For CPU mode, use up to 75% of available cores, capped at 32
        return min(32, max(1, int(multiprocessing.cpu_count() * 0.75)))
    else:
        # For GPU mode, use at least 25% of available cores, still capped at 32
        # This provides enough CPU resources for coordination without excessive overhead
        return min(32, max(1, int(multiprocessing.cpu_count() * 0.25)))

# Initialize with CPU mode default (will be updated in main if needed)
DEFAULT_THREADS = get_default_threads('cpu')

def get_est_time():
    """Get current time in EST timezone with timestamp format"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    return now.strftime('%Y-%m-%d %H:%M:%S EST')

def tprint(message):
    """Print with timestamp in EST"""
    print(f"[{get_est_time()}] {message}")

def analyze_teams(teamsvecs, n_jobs=DEFAULT_THREADS):
    """
    Analyze teams data and generate statistics
    
    Args:
        teamsvecs: Dictionary with team data
        n_jobs: Number of parallel jobs to use
        
    Returns:
        Dictionary with team statistics
    """
    # Try to set NumPy thread count for parallel operations
    try:
        old_threads = np.get_num_threads()
        np.set_num_threads(n_jobs)
        tprint(f"Using {n_jobs} threads for analysis (previous NumPy setting: {old_threads})")
    except Exception as e:
        # Fall back to environment variable for older NumPy versions
        import os
        os.environ["OMP_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)
        tprint(f"Set OMP_NUM_THREADS={n_jobs} for parallel processing")
    
    # Initialize counters
    num_teams = teamsvecs['skill'].shape[0]
    num_skills = teamsvecs['skill'].shape[1]
    num_experts = teamsvecs['member'].shape[1]
    zero_skill_teams = 0
    zero_expert_teams = 0
    max_skills = 0
    min_skills = float('inf')
    max_experts = 0
    min_experts = float('inf')
    dup_teams = 0
    
    # We'll work with sparse matrices directly instead of converting to dense
    tprint("Processing data in sparse format to save memory...")
    skill_matrix = teamsvecs['skill']
    member_matrix = teamsvecs['member']
    
    # Count how many teams each expert is in - can be done on sparse matrix
    tprint("Counting team participation per expert...")
    expert_team_counts = np.array(member_matrix.sum(axis=0)).flatten()
    min_exp_team = int(np.min(expert_team_counts))
    max_exp_team = int(np.max(expert_team_counts))
    
    # Get lists of experts with min/max team participation
    min_team_experts_list = [i for i, count in enumerate(expert_team_counts) if count == min_exp_team]
    max_team_experts_list = [i for i, count in enumerate(expert_team_counts) if count == max_exp_team]
    
    # Analyze each team in parallel batches
    tprint("Analyzing team details in parallel...")
    
    skills_per_team = []
    experts_per_team = []
    skill_indices_per_team = []
    expert_indices_per_team = []
    seen_configs = {}
    
    # Process in batches to save memory
    batch_size = 10000  # Adjust based on available memory
    num_batches = int(np.ceil(num_teams / batch_size))
    
    # Use parallel processing for team analysis
    if n_jobs > 1 and num_teams > 10000:  # Only parallelize for larger datasets
        tprint(f"Processing {num_teams} teams in {num_batches} batches using {n_jobs} parallel jobs...")
        
        # Create batches for parallel processing
        batch_results = Parallel(n_jobs=min(n_jobs, num_batches))(
            delayed(process_team_batch)(
                batch_idx * batch_size,
                min((batch_idx + 1) * batch_size, num_teams),
                skill_matrix,
                member_matrix
            ) for batch_idx in tqdm(range(num_batches), desc="Processing team batches", ncols=100)
        )
        
        # Merge results from all batches
        for (
            skills_batch, experts_batch, skill_indices_batch, expert_indices_batch,
            zero_skill_batch, zero_expert_batch, max_skills_batch, min_skills_batch,
            max_experts_batch, min_experts_batch, seen_configs_batch, dup_teams_batch
        ) in batch_results:
            skills_per_team.extend(skills_batch)
            experts_per_team.extend(experts_batch)
            skill_indices_per_team.extend(skill_indices_batch)
            expert_indices_per_team.extend(expert_indices_batch)
            zero_skill_teams += zero_skill_batch
            zero_expert_teams += zero_expert_batch
            max_skills = max(max_skills, max_skills_batch)
            if min_skills_batch != float('inf'):
                min_skills = min(min_skills, min_skills_batch)
            max_experts = max(max_experts, max_experts_batch)
            if min_experts_batch != float('inf'):
                min_experts = min(min_experts, min_experts_batch)
            
            # Merge seen configurations from this batch
            for team_config, team_idx in seen_configs_batch.items():
                if team_config in seen_configs:
                    dup_teams += 1
                else:
                    seen_configs[team_config] = team_idx
            
            # Note: we don't add dup_teams_batch directly since we need to check 
            # for duplicates across batches, not just within each batch
            
        tprint(f"Merged results from {num_batches} batches, found {dup_teams} duplicate teams")
    else:
        # Original sequential processing for smaller datasets
        with tqdm(total=num_teams, desc="Team analysis", mininterval=0.1, unit="teams", ncols=100) as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_teams)
                
                # Get the skill and member data for this batch
                batch_skill_rows = skill_matrix[start_idx:end_idx]
                batch_member_rows = member_matrix[start_idx:end_idx]
                
                # Process each team in the batch
                for i in range(end_idx - start_idx):
                    team_idx = start_idx + i
                    
                    # Get skill and expert data for this team
                    skill_row = batch_skill_rows[i]
                    member_row = batch_member_rows[i]
                    
                    # Count non-zeros in this team
                    num_skills_in_team = skill_row.getnnz()
                    num_experts_in_team = member_row.getnnz()
                    
                    # Store counts
                    skills_per_team.append(num_skills_in_team)
                    experts_per_team.append(num_experts_in_team)
                    
                    # Get non-zero indices for this team
                    skill_indices = tuple(skill_row.nonzero()[1])
                    expert_indices = tuple(member_row.nonzero()[1])
                    
                    # Store indices
                    skill_indices_per_team.append(skill_indices)
                    expert_indices_per_team.append(expert_indices)
                    
                    # Count special cases
                    if num_skills_in_team == 0:
                        zero_skill_teams += 1
                    if num_experts_in_team == 0:
                        zero_expert_teams += 1
                    
                    # Update max/min metrics
                    max_skills = max(max_skills, num_skills_in_team)
                    if num_skills_in_team > 0:  # Only update min if non-zero
                        min_skills = min(min_skills, num_skills_in_team)
                    
                    max_experts = max(max_experts, num_experts_in_team)
                    if num_experts_in_team > 0:  # Only update min if non-zero
                        min_experts = min(min_experts, num_experts_in_team)
                    
                    # Handle team configuration tracking for duplicates
                    team_config = (skill_indices, expert_indices)
                    
                    if team_config in seen_configs:
                        dup_teams += 1
                    else:
                        seen_configs[team_config] = team_idx
                    
                    # Update progress bar
                    pbar.update(1)
    
    # Calculate number of unique team configurations
    tprint("Calculating team statistics...")
    sys.stdout.flush()
    unique_team_configs = len(seen_configs)
    tprint(f"Found {unique_team_configs} unique team configurations out of {num_teams} teams")
    
    # Second pass: assign duplicate indices
    tprint("Processing team details (second pass)...")
    sys.stdout.flush()  # Ensure the message appears immediately
    
    dup_indices = ['0'] * num_teams  # Pre-allocate
    
    with tqdm(total=num_teams, desc="Team analysis (pass 2)", mininterval=0.1, unit="teams", ncols=100) as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_teams)
            batch_size_actual = end_idx - start_idx
            
            # Get the skill and member data for this batch
            batch_skill_rows = skill_matrix[start_idx:end_idx]
            batch_member_rows = member_matrix[start_idx:end_idx]
            
            # Process each team in the batch
            for i in range(batch_size_actual):
                team_idx = start_idx + i
                
                # Get the skill and member indices for this team
                skill_row = batch_skill_rows[i]
                member_row = batch_member_rows[i]
                
                skill_indices = tuple(skill_row.nonzero()[1])
                expert_indices = tuple(member_row.nonzero()[1])
                
                team_config = (skill_indices, expert_indices)
                
                if team_config in seen_configs and seen_configs[team_config] != team_idx:
                    dup_indices[team_idx] = str(seen_configs[team_config])
                
                # Update progress bar
                pbar.update(1)
            
            # Clear memory after processing each batch
            del batch_skill_rows
            del batch_member_rows
    
    # If no teams were processed, set min values to 0
    if min_skills == float('inf'):
        min_skills = 0
    if min_experts == float('inf'):
        min_experts = 0
    
    return {
        'num_teams': num_teams,
        'num_skills': num_skills,
        'num_experts': num_experts,
        'max_skills': max_skills,
        'min_skills': min_skills,
        'max_experts': max_experts,
        'min_experts': min_experts,
        'zero_skill_teams': zero_skill_teams,
        'zero_expert_teams': zero_expert_teams,
        'unique_team_configs': unique_team_configs,
        'dup_teams': dup_teams,
        'skills_per_team': skills_per_team,
        'experts_per_team': experts_per_team,
        'skill_indices_per_team': skill_indices_per_team,
        'expert_indices_per_team': expert_indices_per_team,
        'dup_indices': dup_indices,
        'min_exp_team': min_exp_team,
        'max_exp_team': max_exp_team,
        'expert_team_counts': expert_team_counts,
        'min_team_experts_list': min_team_experts_list,
        'max_team_experts_list': max_team_experts_list
    }

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
    batch_skill_rows = skill_matrix[start_idx:end_idx]
    batch_member_rows = member_matrix[start_idx:end_idx]
    
    skills_per_team_batch = []
    experts_per_team_batch = []
    skill_indices_per_team_batch = []
    expert_indices_per_team_batch = []
    zero_skill_teams_batch = 0
    zero_expert_teams_batch = 0
    max_skills_batch = 0
    min_skills_batch = float('inf')
    max_experts_batch = 0
    min_experts_batch = float('inf')
    
    # Track unique configurations in this batch
    seen_configs_batch = {}
    dup_teams_batch = 0
    
    for i in range(end_idx - start_idx):
        team_idx = start_idx + i
        
        skill_row = batch_skill_rows[i]
        member_row = batch_member_rows[i]
        
        skill_indices = tuple(skill_row.nonzero()[1])
        expert_indices = tuple(member_row.nonzero()[1])
        
        num_skills_in_team = len(skill_indices)
        num_experts_in_team = len(expert_indices)
        
        skills_per_team_batch.append(num_skills_in_team)
        experts_per_team_batch.append(num_experts_in_team)
        skill_indices_per_team_batch.append(skill_indices)
        expert_indices_per_team_batch.append(expert_indices)
        
        if num_skills_in_team == 0:
            zero_skill_teams_batch += 1
        if num_experts_in_team == 0:
            zero_expert_teams_batch += 1
        
        max_skills_batch = max(max_skills_batch, num_skills_in_team)
        if num_skills_in_team > 0:  # Only update min if non-zero
            min_skills_batch = min(min_skills_batch, num_skills_in_team)
        
        max_experts_batch = max(max_experts_batch, num_experts_in_team)
        if num_experts_in_team > 0:  # Only update min if non-zero
            min_experts_batch = min(min_experts_batch, num_experts_in_team)
        
        # Track team configurations for duplicates within this batch
        team_config = (skill_indices, expert_indices)
        if team_config in seen_configs_batch:
            dup_teams_batch += 1
        else:
            seen_configs_batch[team_config] = team_idx
    
    # Clean up memory
    del batch_skill_rows
    del batch_member_rows
    
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
        seen_configs_batch,  # Return the seen configurations for this batch
        dup_teams_batch  # Return the duplicate count for this batch
    )

def prepare_data_for_gpu(skill_indices_per_team, member_indices_per_team):
    """
    Convert team data to a format suitable for GPU processing
    This is more memory-efficient than creating large dense matrices
    
    Args:
        skill_indices_per_team: List of tuples containing skill indices for each team
        member_indices_per_team: List of tuples containing member indices for each team
        
    Returns:
        tuple: (skill_indices, team_indices_for_skills, member_indices, team_indices_for_members)
    """
    # Prepare data for GPU - skills
    skill_indices = []
    team_indices_for_skills = []
    for team_idx, team_skills in enumerate(skill_indices_per_team):
        for skill_idx in team_skills:
            skill_indices.append(skill_idx)
            team_indices_for_skills.append(team_idx)
    
    # Prepare data for GPU - members
    member_indices = []
    team_indices_for_members = []
    for team_idx, team_members in enumerate(member_indices_per_team):
        for member_idx in team_members:
            member_indices.append(member_idx)
            team_indices_for_members.append(team_idx)
    
    return (
        skill_indices, 
        team_indices_for_skills, 
        member_indices, 
        team_indices_for_members
    )

def generate_distribution_charts(stats, output_dir, dataset_name, n_jobs=DEFAULT_THREADS, use_gpu=False):
    """
    Generate distribution charts for teams statistics
    
    Args:
        stats: Dictionary with team statistics
        output_dir: Directory to save the charts
        dataset_name: Name of the dataset
        n_jobs: Number of parallel jobs to use for matplotlib
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        Dictionary with chart file paths
    """
    # Check if GPU mode is requested and available
    if use_gpu:
        gpu_available = try_import_gpu_libs()
        if gpu_available:
            import cupy as cp
            tprint(f"Using GPU acceleration for distribution charts with {n_jobs} CPU threads for coordination")
            
            # Prepare data for GPU processing
            tprint("Preparing data for GPU processing...")
            gpu_prep_start = time.time()
            skill_indices, team_indices_for_skills, member_indices, team_indices_for_members = prepare_data_for_gpu(
                stats['skill_indices_per_team'], 
                stats['expert_indices_per_team']
            )
            
            # Convert to GPU arrays
            skill_indices_gpu = cp.array(skill_indices, dtype=cp.int32)
            team_indices_for_skills_gpu = cp.array(team_indices_for_skills, dtype=cp.int32)
            member_indices_gpu = cp.array(member_indices, dtype=cp.int32)
            team_indices_for_members_gpu = cp.array(team_indices_for_members, dtype=cp.int32)
            
            # Save GPU tensors in stats for later use
            stats['_gpu'] = {
                'skill_indices': skill_indices_gpu,
                'team_indices_for_skills': team_indices_for_skills_gpu,
                'member_indices': member_indices_gpu,
                'team_indices_for_members': team_indices_for_members_gpu
            }
            
            gpu_prep_time = time.time() - gpu_prep_start
            tprint(f"Data prepared for GPU in {gpu_prep_time:.2f} seconds: {len(skill_indices)} skill-team pairs, {len(member_indices)} member-team pairs")
        else:
            tprint("GPU acceleration requested but not available. Falling back to CPU.")
            use_gpu = False
            
    # Begin timing major chart types
    histogram_start = time.time()
    
    # 1. Skills-to-teams histogram
    tprint("Generating skills-to-teams histogram...")
    plt.figure(figsize=(10, 6))
    
    # Count how many skills are assigned to each number of teams
    # First, count how many teams have each skill
    tprint("Counting teams per skill for histogram...")
    teams_per_skill = []
    
    if use_gpu and CUPY_AVAILABLE:
        # GPU-accelerated version to count teams per skill
        import cupy as cp
        
        # Use pre-computed GPU data if available
        if '_gpu' in stats:
            tprint("Using pre-computed GPU data tensors")
            skill_indices_gpu = stats['_gpu']['skill_indices']
            team_indices_for_skills_gpu = stats['_gpu']['team_indices_for_skills']
        else:
            # Otherwise compute them now
            tprint("Converting team skill data for GPU processing...")
            skill_indices = [skill_idx for team_skills in stats['skill_indices_per_team'] 
                           for skill_idx in team_skills]
            team_indices = [team_idx for team_idx, team_skills in enumerate(stats['skill_indices_per_team']) 
                          for _ in team_skills]
            skill_indices_gpu = cp.array(skill_indices, dtype=cp.int32)
            team_indices_for_skills_gpu = cp.array(team_indices, dtype=cp.int32)
        
        # Count teams per skill using GPU
        tprint("Computing teams per skill on GPU...")
        teams_per_skill_gpu = cp.zeros(stats['num_skills'], dtype=cp.int32)
        
        # Process in batches to avoid GPU memory issues
        batch_size = 1000
        num_batches = (stats['num_skills'] + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Counting teams per skill", ncols=100):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, stats['num_skills'])
            
            for skill_idx in range(start_idx, end_idx):
                # Count teams that have this skill
                skill_mask = (skill_indices_gpu == skill_idx)
                unique_teams = cp.unique(team_indices_for_skills_gpu[skill_mask])
                teams_per_skill_gpu[skill_idx] = len(unique_teams)
        
        # Transfer results back to CPU
        teams_per_skill = cp.asnumpy(teams_per_skill_gpu)
    else:
        # Original CPU version
        for skill_idx in tqdm(range(stats['num_skills']), desc="Counting teams per skill", ncols=100):
            # Count teams that have this skill
            skill_team_count = sum(1 for team_skills in stats['skill_indices_per_team'] if skill_idx in team_skills)
            teams_per_skill.append(skill_team_count)
    
    # Count occurrence of each team count
    if use_gpu and CUPY_AVAILABLE:
        # GPU-accelerated counting of distribution
        import cupy as cp
        
        # Move teams_per_skill to GPU if it's not already there
        if not isinstance(teams_per_skill, cp.ndarray):
            teams_per_skill_gpu = cp.array(teams_per_skill)
        else:
            teams_per_skill_gpu = teams_per_skill
            
        # Get unique values and counts
        x_values_gpu, counts_gpu = cp.unique(teams_per_skill_gpu, return_counts=True)
        
        # Sort by team count
        sort_indices = cp.argsort(x_values_gpu)
        x_values = cp.asnumpy(x_values_gpu[sort_indices])
        y_values = cp.asnumpy(counts_gpu[sort_indices])
    else:
        # Original CPU version
        skill_team_count_distribution = Counter(teams_per_skill)
        
        # Sort by number of teams
        x_values = sorted(skill_team_count_distribution.keys())
        y_values = [skill_team_count_distribution[x] for x in x_values]
    
    # Plot histogram
    plt.bar(x_values, y_values, color='steelblue', alpha=0.7)
    plt.title('Distribution of Skills by Number of Teams (Histogram)', fontsize=14)
    plt.xlabel('Number of Teams', fontsize=12)
    plt.ylabel('Number of Skills', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use log scale if numbers are large
    if max(x_values) > 100 or max(y_values) > 100:
        if max(x_values) > 10:
            plt.xscale('log', base=10)
            plt.xticks([10**i for i in range(int(np.log10(max(x_values)))+2)])
        if max(y_values) > 10:
            plt.yscale('log', base=10)
            plt.yticks([10**i for i in range(int(np.log10(max(y_values)))+2)])
    
    # Annotate if it's long-tailed or uniform
    cv = np.std(teams_per_skill) / np.mean(teams_per_skill) if len(teams_per_skill) > 0 and np.mean(teams_per_skill) > 0 else 0
    distribution_type = "Long-tailed" if cv > 1 else "Relatively Uniform"
    plt.annotate(f"Distribution type: {distribution_type} (CV={cv:.2f})", 
                 xy=(0.5, 0.95), xycoords='axes fraction', 
                 ha='center', va='top', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add GPU info to plot if using GPU
    if use_gpu and CUPY_AVAILABLE:
        plt.figtext(0.02, 0.02, "Generated with GPU acceleration", fontsize=8, color='gray')
    
    # Save the figure
    skills_hist_path = output_dir / 'skills_to_teams_histogram.png'
    plt.tight_layout()
    plt.savefig(skills_hist_path)
    plt.close()
    
    histogram_time = time.time() - histogram_start
    tprint(f"Histogram charts generated in {histogram_time:.2f} seconds")
    
    # Begin timing compact plots
    compact_start = time.time()
    
    # 1c. Skills-to-teams compact plot
    tprint("Generating skills compact plot...")
    plt.figure(figsize=(5, 5))  # Square figure
    
    # Count occurrence of each team count
    with tqdm(total=1, desc="Preparing compact plot data", ncols=100) as pbar:
        skill_team_count_distribution = Counter(teams_per_skill)
        
        # Sort by number of teams
        x_values = sorted(skill_team_count_distribution.keys())
        y_values = [skill_team_count_distribution[x] for x in x_values]
        pbar.update(1)
    
    # Plot with 'x' markers
    plt.scatter(x_values, y_values, marker='x', color='blue', alpha=1.0, s=50)
    
    # Get dataset name - lowercase for consistency with the example
    plot_title = dataset_name.lower()
    
    # Add dataset name to top right
    plt.text(0.85, 0.9, plot_title, transform=plt.gca().transAxes, 
             fontsize=18, ha='center', va='center')
    
    # Set log scales for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels
    plt.xlabel('#teams', fontsize=14, fontweight='bold')
    plt.ylabel('#skills', fontsize=14, fontweight='bold')
    
    # Grid
    plt.grid(True, linestyle='-', alpha=0.3)
    
    # Tight layout and remove extra whitespace
    plt.tight_layout()
    
    # Save the figure
    skills_compact_path = output_dir / 'skills_to_teams_compact.png'
    plt.savefig(skills_compact_path)
    plt.close()
    
    compact_time = time.time() - compact_start
    tprint(f"Compact plots generated in {compact_time:.2f} seconds")
    
    # Begin timing heatmaps
    heatmap_start = time.time()
    tprint("Generating enhanced heatmap visualizations...")
    
    # 3. New Heatmap Visualizations (using hexbin plot)
    tprint("Generating enhanced heatmap visualizations...")
    
    # 3a. Skills-to-teams hexbin heatmap
    max_skills_to_show = min(stats['num_skills'], 100)
    team_indices = list(range(min(100, stats['num_teams'])))
    
    # Create data for hexbin
    x_points = []  # Team indices
    y_points = []  # Skill indices
    
    # For each skill, find which teams have it
    with tqdm(total=max_skills_to_show, desc="Generating skills heatmap data", ncols=100) as pbar:
        for skill_idx in range(min(max_skills_to_show, stats['num_skills'])):
            for team_idx in team_indices:
                if skill_idx in stats['skill_indices_per_team'][team_idx]:
                    x_points.append(team_idx)
                    y_points.append(skill_idx)
            pbar.update(1)
    
    # Create hexbin plot
    tprint(f"Plotting skills heatmap with {len(x_points)} data points...")
    plt.figure(figsize=(12, 8))
    if x_points and y_points:  # Only create hexbin if we have data points
        # Use a colorful hexbin plot
        hb = plt.hexbin(x_points, y_points, gridsize=20, cmap='viridis', mincnt=1)
        plt.colorbar(hb, label='Concentration')
    
    plt.title(f'Skills-Teams Relationship Heatmap ({dataset_name})', fontsize=16)
    plt.xlabel('Team Index', fontsize=14)
    plt.ylabel('Skill Index', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    skills_heatmap_path = output_dir / 'skills_to_teams_heatmap.png'
    plt.savefig(skills_heatmap_path)
    plt.close()
    
    heatmap_time = time.time() - heatmap_start
    tprint(f"Heatmaps generated in {heatmap_time:.2f} seconds")
    
    # We need to ensure similar code for experts is also present
    # Experts histogram
    experts_hist_start = time.time()
    
    # 2. Experts-to-teams histogram
    tprint("Generating experts-to-teams histogram...")
    plt.figure(figsize=(10, 6))
    
    # Count how many teams each expert is in (already computed in analyze_teams)
    teams_per_expert = stats['expert_team_counts']
    
    # Get most active experts for later
    tprint("Finding most active experts...")
    
    # Sort experts by activity (number of teams they're in)
    with tqdm(total=1, desc="Sorting experts by activity", ncols=100) as pbar:
        expert_team_count_pairs = [(i, count) for i, count in enumerate(teams_per_expert)]
        expert_team_count_pairs.sort(key=lambda x: x[1], reverse=True)
        most_active_experts = [idx for idx, _ in expert_team_count_pairs]
        pbar.update(1)
    
    # Count teams per expert distribution
    if use_gpu and CUPY_AVAILABLE:
        # GPU-accelerated counting of distribution
        import cupy as cp
        
        # Move teams_per_expert to GPU
        teams_per_expert_gpu = cp.array(teams_per_expert)
            
        # Get unique values and counts
        x_values_gpu, counts_gpu = cp.unique(teams_per_expert_gpu, return_counts=True)
        
        # Sort by team count
        sort_indices = cp.argsort(x_values_gpu)
        x_values = cp.asnumpy(x_values_gpu[sort_indices])
        y_values = cp.asnumpy(counts_gpu[sort_indices])
    else:
        # Original CPU version
        expert_team_count_distribution = Counter(teams_per_expert)
        
        # Sort by number of teams
        x_values = sorted(expert_team_count_distribution.keys())
        y_values = [expert_team_count_distribution[x] for x in x_values]
    
    # Plot histogram
    plt.bar(x_values, y_values, color='firebrick', alpha=0.7)
    plt.title('Distribution of Experts by Number of Teams (Histogram)', fontsize=14)
    plt.xlabel('Number of Teams', fontsize=12)
    plt.ylabel('Number of Experts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use log scale if numbers are large
    if max(x_values) > 100 or max(y_values) > 100:
        if max(x_values) > 10:
            plt.xscale('log', base=10)
            plt.xticks([10**i for i in range(int(np.log10(max(x_values)))+2)])
        if max(y_values) > 10:
            plt.yscale('log', base=10)
            plt.yticks([10**i for i in range(int(np.log10(max(y_values)))+2)])
    
    # Annotate if it's long-tailed or uniform
    cv = np.std(teams_per_expert) / np.mean(teams_per_expert) if len(teams_per_expert) > 0 and np.mean(teams_per_expert) > 0 else 0
    distribution_type = "Long-tailed" if cv > 1 else "Relatively Uniform"
    plt.annotate(f"Distribution type: {distribution_type} (CV={cv:.2f})", 
                 xy=(0.5, 0.95), xycoords='axes fraction', 
                 ha='center', va='top', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add GPU info to plot if using GPU
    if use_gpu and CUPY_AVAILABLE:
        plt.figtext(0.02, 0.02, "Generated with GPU acceleration", fontsize=8, color='gray')
    
    # Save the figure
    experts_hist_path = output_dir / 'experts_to_teams_histogram.png'
    plt.tight_layout()
    plt.savefig(experts_hist_path)
    plt.close()
    
    # Experts compact plot
    tprint("Generating experts compact plot...")
    plt.figure(figsize=(5, 5))  # Square figure
    
    # Count occurrence of each team count
    with tqdm(total=1, desc="Preparing experts compact plot data", ncols=100) as pbar:
        expert_team_count_distribution = Counter(teams_per_expert)
        
        # Sort by number of teams
        x_values = sorted(expert_team_count_distribution.keys())
        y_values = [expert_team_count_distribution[x] for x in x_values]
        pbar.update(1)
    
    # Plot with 'x' markers
    plt.scatter(x_values, y_values, marker='x', color='blue', alpha=1.0, s=50)
    
    # Add dataset name to top right
    plt.text(0.85, 0.9, plot_title, transform=plt.gca().transAxes, 
             fontsize=18, ha='center', va='center')
    
    # Set log scales for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels
    plt.xlabel('#teams', fontsize=14, fontweight='bold')
    plt.ylabel('#experts', fontsize=14, fontweight='bold')
    
    # Grid
    plt.grid(True, linestyle='-', alpha=0.3)
    
    # Tight layout and remove extra whitespace
    plt.tight_layout()
    
    # Save the figure
    experts_compact_path = output_dir / 'experts_to_teams_compact.png'
    plt.savefig(experts_compact_path)
    plt.close()
    
    # Experts heatmap
    tprint("Generating experts-to-teams heatmap...")
    
    # 3b. Experts-to-teams hexbin heatmap
    max_experts_to_show = min(stats['num_experts'], 100)
    team_indices = list(range(min(100, stats['num_teams'])))
    
    # Create data for hexbin
    x_points = []  # Team indices
    y_points = []  # Expert indices
    
    # For each expert (among the most active ones), find which teams they're in
    with tqdm(total=max_experts_to_show, desc="Generating experts heatmap data", ncols=100) as pbar:
        for i, expert_idx in enumerate(most_active_experts[:max_experts_to_show]):
            for team_idx in team_indices:
                if expert_idx in stats['expert_indices_per_team'][team_idx]:
                    x_points.append(team_idx)
                    y_points.append(i)  # Use the position in the most_active_experts list for better visualization
            pbar.update(1)
    
    # Create hexbin plot
    tprint(f"Plotting experts heatmap with {len(x_points)} data points...")
    plt.figure(figsize=(12, 8))
    if x_points and y_points:  # Only create hexbin if we have data points
        # Use a colorful hexbin plot
        hb = plt.hexbin(x_points, y_points, gridsize=20, cmap='plasma', mincnt=1)
        plt.colorbar(hb, label='Concentration')
    
    plt.title(f'Experts-Teams Relationship Heatmap ({dataset_name})', fontsize=16)
    plt.xlabel('Team Index', fontsize=14)
    plt.ylabel('Expert Index (by activity rank)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    experts_heatmap_path = output_dir / 'experts_to_teams_heatmap.png'
    plt.savefig(experts_heatmap_path)
    plt.close()
    
    experts_hist_time = time.time() - experts_hist_start
    tprint(f"Experts charts generated in {experts_hist_time:.2f} seconds")
    
    # Return paths to all charts
    chart_paths = {
        'skills_histogram': str(skills_hist_path),
        'skills_compact': str(skills_compact_path),
        'skills_heatmap': str(skills_heatmap_path),
        'experts_histogram': str(experts_hist_path),
        'experts_compact': str(experts_compact_path),
        'experts_heatmap': str(experts_heatmap_path)
    }
    
    # Add relative paths for the markdown report
    try:
        relative_chart_paths = {}
        for key, path in chart_paths.items():
            rel_path = Path(path).relative_to(output_dir.parent)
            relative_chart_paths[key] = str(rel_path)
        
        # Report total chart generation time
        total_chart_time = time.time() - chart_start_time if 'chart_start_time' in locals() else 0
        tprint(f"All chart generation completed in {total_chart_time:.2f} seconds")
        
        return relative_chart_paths
    except ValueError:
        # If relative_to fails, return the absolute paths
        return chart_paths

def generate_markdown_report(stats, dataset_name, filter_info, output_dir, removed_items=None):
    """
    Generate markdown report with statistics and visualizations
    
    Args:
        stats: Dictionary containing team statistics
        dataset_name: Name of the dataset
        filter_info: Dictionary with filtering information
        output_dir: Directory to save the markdown file
        removed_items: List of items removed during filtering
    """
    # No need to create the directory here - it will be created by the stats-reports directory
    
    # Define terms based on dataset
    if dataset_name.upper() == 'GITH':
        team_term = "Repositories"
        expert_term = "Contributors"
        skill_term = "Languages"
    elif dataset_name.upper() == 'DBLP':
        team_term = "Papers"
        expert_term = "Authors"
        skill_term = "Keywords"
    elif dataset_name.upper() == 'USPT':
        team_term = "Patents"
        expert_term = "Inventors"
        skill_term = "CPC Codes"
    elif dataset_name.upper() == 'IMDB':
        team_term = "Movies"
        expert_term = "Actors"
        skill_term = "Genres"
    elif dataset_name.upper() == 'CF9':
        team_term = "Meals"
        expert_term = "Ingredients"
        skill_term = "Flavor Compounds"  
    else:
        team_term = "Teams"
        expert_term = "Members"
        skill_term = "Skills"
    
    # Calculate additional statistics
    avg_contributors_per_repo = sum(stats['experts_per_team']) / stats['num_teams'] if stats['num_teams'] > 0 else 0
    avg_skills_per_repo = sum(stats['skills_per_team']) / stats['num_teams'] if stats['num_teams'] > 0 else 0
    
    # Calculate average repos per contributor
    if stats['num_experts'] > 0:
        total_expert_participations = sum(stats['expert_team_counts'])
        avg_repos_per_contributor = total_expert_participations / stats['num_experts']
    else:
        avg_repos_per_contributor = 0
    
    # Calculate average skills per contributor (estimate)
    if stats['num_experts'] > 0:
        # This is an approximation - we assume skills are distributed relatively evenly across experts
        # For more precise calculation, we'd need to track which skills each expert has directly
        total_skills_across_teams = sum(stats['skills_per_team'])
        avg_skills_per_contributor = total_skills_across_teams / stats['num_experts']
    else:
        avg_skills_per_contributor = 0
    
    # Add team composition info
    zero_skill_teams = stats['zero_skill_teams']
    zero_expert_teams = stats['zero_expert_teams']
    
    # Calculate teams with neither skills nor experts
    teams_with_neither = sum(1 for i in range(stats['num_teams']) 
                           if i < len(stats['skills_per_team']) and i < len(stats['experts_per_team']) 
                           and stats['skills_per_team'][i] == 0 and stats['experts_per_team'][i] == 0)
    
    # Create markdown content
    md_content = f"""# {dataset_name.upper()} Dataset Statistics

## Dataset Information
- Dataset: {dataset_name.upper()}
- Number of {team_term} (Teams): {stats['num_teams']:,}
- Number of {expert_term} (Experts): {stats['num_experts']:,}
- Number of {skill_term} (Skills): {stats['num_skills']:,}
"""
    
    # Only add Filtering Information section if we have removed items
    if removed_items and len(removed_items) > 0:
        md_content += f"\n## Filtering Information (in order):\n"
        for i, item in enumerate(removed_items, 1):
            md_content += f"{i}. Removed {item}\n"
    
    # Add duplicate teams info
    unique_teams = stats['num_teams'] - stats['dup_teams']
    md_content += f"\n## Duplication Information\n"
    md_content += f"- Unique {team_term} (Teams): {unique_teams:,} ({(unique_teams / stats['num_teams'] * 100) if stats['num_teams'] > 0 else 0:.2f}%)\n"
    md_content += f"- Duplicate {team_term} (Teams): {stats['dup_teams']:,} ({(stats['dup_teams'] / stats['num_teams'] * 100) if stats['num_teams'] > 0 else 0:.2f}%)\n"
    
    # Add note about duplicate counts when we have filter info showing duplicates were removed
    if removed_items and any("Duplicate teams:" in item for item in removed_items) and stats['dup_teams'] == 0:
        removed_count = next((int(item.split(':')[1].strip().replace(',', '')) for item in removed_items if "Duplicate teams:" in item), 0)
        if removed_count > 0:
            md_content += f"\n**Note:** {removed_count:,} duplicate teams were already removed by data-filter.py. "
            md_content += f"The count above shows duplicates found in the current dataset (after filtering).\n"
    
    # Add team composition info
    zero_skill_teams = stats['zero_skill_teams']
    zero_expert_teams = stats['zero_expert_teams']
    
    md_content += f"""
## Composition Information
| Statistic | Value |
|-----------|-------|
| #{team_term} (Teams) w/o {expert_term} (Experts) | {zero_expert_teams:,} |
| #{team_term} (Teams) w/o {skill_term} (Skills) | {zero_skill_teams:,} |
| #{team_term} (Teams) w/o both {expert_term} and {skill_term} | {teams_with_neither:,} |
| Avg #{expert_term} (Experts) per {team_term[:-1]} (Team) | {avg_contributors_per_repo:.5f} |
| Avg #{skill_term} (Skills) per {team_term[:-1]} (Team) | {avg_skills_per_repo:.5f} |
| Avg #{team_term} (Teams) per {expert_term[:-1]} (Expert) | {avg_repos_per_contributor:.5f} |
| Avg #{skill_term} (Skills) per {expert_term[:-1]} (Expert) | {avg_skills_per_contributor:.5f} |

## Distribution Information

### {skill_term} (Skills) to {team_term} (Teams) Distribution
- Min {skill_term.lower()} (skills) per {team_term[:-1].lower()} (team): {stats['min_skills']}
- Max {skill_term.lower()} (skills) per {team_term[:-1].lower()} (team): {stats['max_skills']}
- {team_term} (Teams) with min {skill_term.lower()} (skills): {sum(1 for x in stats['skills_per_team'] if x == stats['min_skills'])} ({(sum(1 for x in stats['skills_per_team'] if x == stats['min_skills']) / stats['num_teams'] * 100):.2f}%)
- {team_term} (Teams) with max {skill_term.lower()} (skills): {sum(1 for x in stats['skills_per_team'] if x == stats['max_skills'])} ({(sum(1 for x in stats['skills_per_team'] if x == stats['max_skills']) / stats['num_teams'] * 100):.2f}%)

### {expert_term} (Experts) to {team_term} (Teams) Distribution
- Min {team_term.lower()} (teams) per {expert_term[:-1].lower()} (expert): {stats['min_exp_team']}
- Max {team_term.lower()} (teams) per {expert_term[:-1].lower()} (expert): {stats['max_exp_team']}
- {expert_term} (Experts) in min {team_term.lower()} (teams): {len(stats['min_team_experts_list'])} ({(len(stats['min_team_experts_list']) / stats['num_experts'] * 100):.2f}%)
- {expert_term} (Experts) in max {team_term.lower()} (teams): {len(stats['max_team_experts_list'])} ({(len(stats['max_team_experts_list']) / stats['num_experts'] * 100):.2f}%)

## Visualization
Please see the accompanying distribution charts for visual representation of data distributions.

### {skill_term} (Skills) to {team_term} (Teams) Visualizations
| Chart Type | File Name | Description |
|------------|-----------|-------------|
| Histogram | `skills_to_teams_histogram.png` | Shows the distribution of {skill_term.lower()} (skills) by the number of {team_term.lower()} (teams) |
| Compact Plot | `skills_to_teams_compact.png` | Compact log-scale visualization of {skill_term.lower()} (skills) distribution by {team_term.lower()} (teams) count |
| Heatmap | `skills_to_teams_heatmap.png` | Heat visualization showing density of {skill_term.lower()} (skills) across {team_term.lower()} (teams) |

### {expert_term} (Experts) to {team_term} (Teams) Visualizations
| Chart Type | File Name | Description |
|------------|-----------|-------------|
| Histogram | `experts_to_teams_histogram.png` | Shows the distribution of {expert_term.lower()} (experts) by the number of {team_term.lower()} (teams) |
| Compact Plot | `experts_to_teams_compact.png` | Compact log-scale visualization of {expert_term.lower()} (experts) distribution by {team_term.lower()} (teams) count |
| Heatmap | `experts_to_teams_heatmap.png` | Heat visualization showing density of {expert_term.lower()} (experts) across {team_term.lower()} (teams) |
"""
    
    # Write to file
    md_path = output_dir / f'{dataset_name.lower()}_statistics.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    return md_path

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
    
    tprint(f"Found {duplicate_count} duplicates using data-filter.py approach")
    return duplicate_count

def parse_gpu_mode(mode_arg):
    """
    Parse the GPU mode argument to extract device indices
    
    Args:
        mode_arg: The mode argument string (e.g., 'cpu', 'gpu', 'gpu=all', 'gpu=2', 'gpu=1,2')
        
    Returns:
        tuple: (mode, device_indices) where mode is 'cpu' or 'gpu' and 
               device_indices is a list of GPU indices, 'all', or 'first'
    """
    if mode_arg == 'cpu':
        return 'cpu', None
    
    # Default case - use the first available GPU
    if mode_arg == 'gpu':
        return 'gpu', 'first'
    
    # Support both equal sign format (preferred) and parentheses format (legacy)
    # Check for gpu=N or gpu=N,M,... format
    if mode_arg.startswith('gpu='):
        # Extract device indices
        devices_str = mode_arg[4:]  # Remove 'gpu='
        
        # Special case for all GPUs
        if devices_str.lower() == 'all':
            return 'gpu', 'all'
            
        try:
            # Parse comma-separated indices
            devices = [int(idx.strip()) for idx in devices_str.split(',')]
            return 'gpu', devices
        except ValueError:
            tprint(f"Warning: Invalid GPU device specification '{mode_arg}'. Using first available GPU.")
            return 'gpu', 'first'
    
    # Legacy support for gpu(N) or gpu(N,M,...) format
    if mode_arg.startswith('gpu(') and mode_arg.endswith(')'):
        # Extract device indices
        devices_str = mode_arg[4:-1]  # Remove 'gpu(' and ')'
        
        # Special case for all GPUs
        if devices_str.lower() == 'all':
            return 'gpu', 'all'
            
        try:
            # Parse comma-separated indices
            devices = [int(idx.strip()) for idx in devices_str.split(',')]
            return 'gpu', devices
        except ValueError:
            tprint(f"Warning: Invalid GPU device specification '{mode_arg}'. Using first available GPU.")
            return 'gpu', 'first'
    
    # If we get here, the format wasn't recognized
    tprint(f"Warning: Unrecognized mode '{mode_arg}'. Falling back to CPU mode.")
    return 'cpu', None

def main():
    # Start the timer
    start_time = time.time()
    
    tprint("Starting data reports process...")
    
    parser = argparse.ArgumentParser(
        description="Generate reports and visualizations for teams data."
    )
    
    # Required arguments
    parser.add_argument('-i', '--input-path', type=str, required=True,
                       help='Path to the input file or directory')
    parser.add_argument('-d', '--dataset', type=str, required=False,
                       help='Dataset name (gith, dblp, etc.)')
    
    # Optional arguments
    parser.add_argument('-o', '--output-dir', type=str, required=False,
                      help='Output directory for reports (default: derived from input path)')
    
    parser.add_argument('-mode', '--mode', type=str, default='cpu',
                      help='Processing mode: cpu (default), gpu (first GPU), gpu=all (all GPUs), gpu=N (specific GPU), gpu=N,M (multiple GPUs)')
    
    # Update default threads based on mode
    global DEFAULT_THREADS
    DEFAULT_THREADS = get_default_threads('cpu')  # Default to CPU mode initially
    
    parser.add_argument('-t', '--threads', type=int, default=DEFAULT_THREADS,
                      help=f'Number of parallel threads to use (default: auto-detected based on mode)')
    
    parser.add_argument('-fs', '--filter-stats', type=str, default='filter_stats.txt',
                      help='Path to filter stats file (default: filter_stats.txt in the same directory as the input file)')
    
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    args = parser.parse_args()
    
    # Parse the mode to extract GPU device indices if specified
    mode, device_indices = parse_gpu_mode(args.mode)
    
    # If GPU devices were specified, set them globally
    if mode == 'gpu':
        global SELECTED_GPU_DEVICES
        SELECTED_GPU_DEVICES = device_indices
    
    # Update default threads based on selected mode if user didn't specify
    if args.threads == DEFAULT_THREADS:  # Only if user didn't override
        DEFAULT_THREADS = get_default_threads(mode)
        args.threads = DEFAULT_THREADS
        if mode == 'cpu':
            tprint(f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {args.threads} threads ({int(args.threads/multiprocessing.cpu_count()*100)}% of CPU)")
        else:
            if device_indices == 'first':
                tprint(f"Using {args.threads} CPU threads for GPU-accelerated mode with first available GPU")
            elif device_indices == 'all':
                tprint(f"Using {args.threads} CPU threads for GPU-accelerated mode with all available GPUs")
            elif isinstance(device_indices, list):
                gpu_str = ", ".join(map(str, device_indices))
                tprint(f"Using {args.threads} CPU threads for GPU-accelerated mode with GPU(s): {gpu_str}")
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
        teamsvecs_path = input_path / 'teamsvecs.pkl'
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
        if 'gith' in path_str:
            dataset_name = 'gith'
        elif 'dblp' in path_str:
            dataset_name = 'dblp'
        elif 'uspt' in path_str:
            dataset_name = 'uspt'
        elif 'imdb' in path_str:
            dataset_name = 'imdb'
        elif 'cf9' in path_str:
            dataset_name = 'cf9'
        else:
            dataset_name = 'unknown'
        
        tprint(f"Inferred dataset name: {dataset_name}")
    else:
        dataset_name = args.dataset.lower()
    
    # Load the data
    tprint(f"Loading data from {input_path}...")
    with tqdm(total=1, desc="Loading data", ncols=100) as pbar:
        try:
            with open(input_path, 'rb') as f:
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
            with open(filter_stats_path, 'r') as f:
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
                    if ':' in line and not in_removed_section:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                filter_info[key] = value
                                
        except Exception as e:
            tprint(f"Warning: Could not parse filter stats file: {e}")
    
    # Add debug flag to compare duplicate detection with filter.py
    if args.debug and 'Duplicate teams' in filter_info:
        tprint(f"\nDEBUG: Filter stats reported {filter_info['Duplicate teams']} duplicates")
        tprint("DEBUG: Will compare with data-reports.py duplicate detection method")

    tprint("Analyzing teams data...")
    stats = analyze_teams(teamsvecs, n_jobs=args.threads)

    # Add diagnostic duplicate detection when in debug mode
    if args.debug:
        tprint("\nRunning diagnostic duplicate detection...")
        filter_style_dup_count = analyze_duplicate_detection(
            teamsvecs['skill'], teamsvecs['member'])
        tprint(f"Filter-style duplicate count: {filter_style_dup_count}")
        tprint(f"Reports-style duplicate count: {stats['dup_teams']}")
    
    # Compare duplicate counts if debugging
    if args.debug and 'Duplicate teams' in filter_info:
        filter_dup_count = int(filter_info['Duplicate teams'].replace(',', ''))
        reports_dup_count = stats['dup_teams']
        tprint(f"\nDEBUG: Duplicate team comparison:")
        tprint(f"  - data-filter.py found: {filter_dup_count} duplicates")
        tprint(f"  - data-reports.py found: {reports_dup_count} duplicates")
        tprint(f"  - Difference: {abs(filter_dup_count - reports_dup_count)} teams")
        if filter_dup_count != reports_dup_count:
            tprint(f"  - Possible reasons for difference:")
            tprint(f"    1. Different definition of duplicates (filter looks at both experts & skills, reports might have additional criteria)")
            tprint(f"    2. Different ordering/processing of teams during duplicate detection")
            tprint(f"    3. Filter removes duplicates before analysis, reports counts after they're already removed")

    # Create output file in the same directory as the input file
    input_filename = input_path.name
    output_suffix = f'_{input_filename.replace(".pkl", "")}' if input_filename != 'teamsvecs.pkl' else ''
    
    # Create stats-reports directory next to the input file
    stats_reports_dir = input_path.parent / 'stats-reports'
    stats_reports_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate CSV report
    output_file = stats_reports_dir / f'{dataset_name}_team_stats{output_suffix}.csv'
    
    tprint(f"Writing report to {output_file}...")
    sys.stdout.flush()
    with tqdm(total=1, desc="Writing report", ncols=100) as pbar:
        with open(output_file, 'w') as f:
            # Basic stats with dataset name
            f.write(',dataset,#teams,#skills,#experts\n')
            f.write(f',{dataset_name.upper()},{stats["num_teams"]},{stats["num_skills"]},{stats["num_experts"]}\n')
            f.write(',,,,\n')
            
            # Zero teams and duplicates stats
            f.write(',#zero_skill_teams,#zero_expert_teams,#dup_teams,#unique_teams\n')
            zero_skill_percent = (stats['zero_skill_teams'] / stats['num_teams']) * 100
            zero_expert_percent = (stats['zero_expert_teams'] / stats['num_teams']) * 100
            dup_teams_percent = (stats['dup_teams'] / stats['num_teams']) * 100
            unique_teams = stats['unique_team_configs']
            unique_teams_percent = (unique_teams / stats['num_teams']) * 100
            
            f.write(f',{stats["zero_skill_teams"]} ({zero_skill_percent:.1f}% of teams),' +
                    f'{stats["zero_expert_teams"]} ({zero_expert_percent:.1f}% of teams),' +
                    f'{stats["dup_teams"]} ({dup_teams_percent:.1f}% of teams),' +
                    f'{unique_teams} ({unique_teams_percent:.1f}% of teams)\n')
            f.write(',,,,\n')
            
            # Team size stats
            f.write(',#min_team_experts,#max_team_experts,#min_team_skill,#max_team_skill\n')
            
            # Count teams with min/max experts/skills
            min_expert_teams = sum(1 for x in stats['expert_team_counts'] if x == stats['min_exp_team'])
            max_expert_teams = sum(1 for x in stats['expert_team_counts'] if x == stats['max_exp_team'])
            min_skill_teams = sum(1 for x in stats['skills_per_team'] if x == stats['min_skills'])
            max_skill_teams = sum(1 for x in stats['skills_per_team'] if x == stats['max_skills'])
            
            min_expert_percent = (min_expert_teams / stats['num_teams']) * 100
            max_expert_percent = (max_expert_teams / stats['num_teams']) * 100
            min_skill_percent = (min_skill_teams / stats['num_teams']) * 100
            max_skill_percent = (max_skill_teams / stats['num_teams']) * 100
            
            f.write(f',{stats["min_exp_team"]} ({min_expert_teams} experts~{min_expert_percent:.1f}%),' +
                    f'{stats["max_exp_team"]} ({max_expert_teams} experts~{max_expert_percent:.1f}%),' +
                    f'{stats["min_skills"]} ({min_skill_teams} teams~{min_skill_percent:.1f}%),' +
                    f'{stats["max_skills"]} ({max_skill_teams} teams~{max_skill_percent:.1f}%)\n')
            f.write(',,,\n')
            
            # Expert participation stats
            f.write(',#min_exp_team,#max_exp_team\n')
            
            # Count experts with min/max team participation
            min_team_experts = sum(1 for x in stats['expert_team_counts'] if x == stats['min_exp_team'])
            max_team_experts = sum(1 for x in stats['expert_team_counts'] if x == stats['max_exp_team'])
            
            min_exp_percent = (min_team_experts / stats['num_experts']) * 100
            max_exp_percent = (max_team_experts / stats['num_experts']) * 100
            
            # Count teams with min/max experts/skills
            min_expert_teams = sum(1 for x in stats['experts_per_team'] if x == stats['min_experts'])
            max_expert_teams = sum(1 for x in stats['experts_per_team'] if x == stats['max_experts'])
            min_skill_teams = sum(1 for x in stats['skills_per_team'] if x == stats['min_skills'])
            max_skill_teams = sum(1 for x in stats['skills_per_team'] if x == stats['max_skills'])
            
            min_expert_percent = (min_expert_teams / stats['num_teams']) * 100
            max_expert_percent = (max_expert_teams / stats['num_teams']) * 100
            min_skill_percent = (min_skill_teams / stats['num_teams']) * 100
            max_skill_percent = (max_skill_teams / stats['num_teams']) * 100
            
            f.write(f',{stats["min_experts"]} ({min_expert_teams} teams~{min_expert_percent:.1f}%),' +
                    f'{stats["max_experts"]} ({max_expert_teams} teams~{max_expert_percent:.1f}%),' +
                    f'{stats["min_skills"]} ({min_skill_teams} teams~{min_skill_percent:.1f}%),' +
                    f'{stats["max_skills"]} ({max_skill_teams} teams~{max_skill_percent:.1f}%)\n')
            
            # For detailed section
            f.write(f'\n### Expert Participation\n\n')
            f.write(f'- Min teams per expert: {stats["min_exp_team"]} ({min_team_experts} experts ~ {min_exp_percent:.1f}%)\n')
            f.write(f'- Max teams per expert: {stats["max_exp_team"]} ({max_team_experts} experts ~ {max_exp_percent:.1f}%)\n\n')
            
            # Write header for detailed stats
            f.write('dup_index,#skills,#experts,skills,experts\n')
            
            # Write detailed stats for each team
            tprint("Writing detailed team statistics...")
            sys.stdout.flush()
            with tqdm(total=len(stats['skills_per_team']), desc="Writing team details", mininterval=0.1, unit="teams", ncols=100) as team_pbar:
                for i in range(len(stats['skills_per_team'])):
                    # Add 1 to indices to start from 1 instead of 0
                    skill_indices = [f's{idx+1}' for idx in stats['skill_indices_per_team'][i]]
                    expert_indices = [f'm{idx+1}' for idx in stats['expert_indices_per_team'][i]]
                    
                    f.write(f'{stats["dup_indices"][i]},{stats["skills_per_team"][i]},{len(expert_indices)},' +
                          f'[{"-".join(skill_indices)}],' +
                          f'[{"-".join(expert_indices)}]\n')
                    
                    # Update progress bar
                    team_pbar.update(1)
                    
                    # For very large datasets, update less frequently to avoid slowdown
                    if i % 1000 == 0:
                        team_pbar.refresh()
        pbar.update(1)
    
    # Generate distribution charts
    tprint("Generating distribution charts...")
    chart_start_time = time.time()
    use_gpu = mode == 'gpu'
    if use_gpu:
        if device_indices == 'first':
            tprint("GPU mode enabled - using first available GPU")
        elif device_indices == 'all':
            tprint("GPU mode enabled - using all available GPUs")
        elif isinstance(device_indices, list):
            gpu_str = ", ".join(map(str, device_indices))
            tprint(f"GPU mode enabled - using specified GPU(s): {gpu_str}")
        else:
            tprint("GPU mode enabled - will try to use GPU acceleration")
    
    charts = generate_distribution_charts(stats, stats_reports_dir, dataset_name, n_jobs=args.threads, use_gpu=use_gpu)
    
    chart_time = time.time() - chart_start_time
    tprint(f"Chart generation completed in {chart_time:.2f} seconds")
    
    # Generate markdown report
    tprint("Generating markdown report...")
    md_file_path = generate_markdown_report(stats, dataset_name, filter_info, stats_reports_dir, removed_items)
    
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
                tprint(f"GPU memory: {free_mem_gb:.1f} GB free / {total_mem_gb:.1f} GB total")
            except Exception as e:
                tprint(f"Could not get GPU memory info: {e}")
        else:
            tprint("GPU acceleration was requested but not available. Used CPU instead.")
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    tprint(f"Total execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")

if __name__ == '__main__':
    main() 