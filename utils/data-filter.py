import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import argparse
import os
import subprocess
from datetime import datetime
import pytz
import time
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from multiprocessing import Manager

# Set default thread count to 75% of available cores, capped at 32
DEFAULT_THREADS = min(32, max(1, int(multiprocessing.cpu_count() * 0.75)))

def get_est_time():
    """Get current time in EST timezone with timestamp format"""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)
    return now.strftime('%Y-%m-%d %H:%M:%S EST')

def tprint(message):
    """Print with timestamp in EST"""
    print(f"[{get_est_time()}] {message}")

def process_batch(batch_idx, start_idx, end_idx, skill_matrix, member_matrix, team_configs, progress_update=None):
    """
    Process a batch of teams to detect duplicates in parallel
    
    Args:
        batch_idx: Current batch index
        start_idx: Start index for this batch
        end_idx: End index for this batch
        skill_matrix: Matrix of team skills
        member_matrix: Matrix of team members
        team_configs: Set of existing team configurations
        progress_update: Function to call for progress updates
        
    Returns:
        Tuple containing (local unique mask, new team configs)
    """
    batch_skill_rows = skill_matrix[start_idx:end_idx]
    batch_member_rows = member_matrix[start_idx:end_idx]
    
    batch_size_actual = end_idx - start_idx
    local_unique_mask = np.ones(batch_size_actual, dtype=bool)
    local_configs = []
    
    for i in range(batch_size_actual):
        team_idx = i  # Local index within the batch
        
        # Get non-zero indices for this team
        skill_row = batch_skill_rows[i]
        member_row = batch_member_rows[i]
        
        skill_indices = tuple(skill_row.nonzero()[1])
        member_indices = tuple(member_row.nonzero()[1])
        
        team_config = (skill_indices, member_indices)
        
        # Check if this team config exists in the shared set
        if team_config in team_configs:
            local_unique_mask[team_idx] = False
        else:
            local_configs.append(team_config)
        
        # Update progress periodically
        if progress_update and (start_idx + i + 1) % 100000 == 0:
            progress_update(start_idx + i + 1, skill_matrix.shape[0])
    
    # Clean up to help with memory management
    del batch_skill_rows
    del batch_member_rows
    
    return local_unique_mask, local_configs

def fix_data(teamsvecs, indexes, remove_duplicates=True, minimum_teams=None, team_size=None, 
             remove_zero_experts_teams=False, remove_zero_skills_teams=False, n_jobs=DEFAULT_THREADS):
    """
    Fix data by removing teams and experts that don't meet criteria
    
    Args:
        teamsvecs: Dictionary with team data
        indexes: Dictionary with team and expert indexes
        remove_duplicates: Whether to remove duplicate teams
        minimum_teams: Minimum number of teams an expert must be in
        team_size: Minimum number of experts a team must have
        remove_zero_experts_teams: Whether to remove teams with no experts
        remove_zero_skills_teams: Whether to remove teams with no skills
        n_jobs: Number of parallel jobs to use
        
    Returns:
        Tuple containing fixed team data, updated indexes, and filter statistics
    """
    # Work with sparse matrices directly to save memory
    skill_matrix = teamsvecs['skill']
    member_matrix = teamsvecs['member']
    
    num_teams = skill_matrix.shape[0]
    num_experts = member_matrix.shape[1]
    
    tprint(f"Initial number of teams: {num_teams}")
    tprint(f"Initial number of experts: {num_experts}")
    
    # Initialize statistics dictionary
    filter_stats = {
        'initial_teams': num_teams,
        'initial_experts': num_experts,
        'initial_skills': skill_matrix.shape[1],
        'zero_expert_teams_only': 0,
        'zero_skill_teams_only': 0, 
        'teams_with_neither': 0,
        'duplicate_teams': 0,
        'teams_below_min_experts': 0,
        'remaining_teams': num_teams
    }
    
    # Initialize cumulative mask to track teams that pass all filters
    cumulative_mask = np.ones(skill_matrix.shape[0], dtype=bool)
    
    # Calculate initial masks
    team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
    team_skill_sums = np.array(skill_matrix.sum(axis=1)).flatten()
    has_experts = team_expert_sums > 0
    has_skills = team_skill_sums > 0
    
    # Count original statistics for reporting
    zero_expert_teams_only = np.sum(~has_experts & has_skills)    # Teams with no experts but have skills
    zero_skill_teams_only = np.sum(has_experts & ~has_skills)     # Teams with no skills but have experts
    teams_with_neither = np.sum(~has_experts & ~has_skills)       # Teams with neither experts nor skills
    
    total_zero_expert_teams = zero_expert_teams_only + teams_with_neither
    total_zero_skill_teams = zero_skill_teams_only + teams_with_neither
    
    filter_stats['zero_expert_teams_only'] = int(zero_expert_teams_only)   # Teams with no experts but have skills
    filter_stats['zero_skill_teams_only'] = int(zero_skill_teams_only)     # Teams with no skills but have experts
    filter_stats['teams_with_neither'] = int(teams_with_neither)           # Teams with neither experts nor skills
    
    # For backward compatibility with any code that uses these
    # These totals include the 'neither' category
    filter_stats['zero_expert_teams'] = int(total_zero_expert_teams)
    filter_stats['zero_skill_teams'] = int(total_zero_skill_teams)
    
    # Set NumPy to use multiple threads for BLAS operations
    try:
        # Try to set the number of threads for NumPy operations
        # This may fail on older NumPy versions
        old_threads = 1  # Default assumption
        
        # Check if get_num_threads exists before calling it
        if hasattr(np, 'get_num_threads'):
            old_threads = np.get_num_threads()
            
        # Try different ways to set thread count based on NumPy version
        if hasattr(np, 'set_num_threads'):
            np.set_num_threads(n_jobs)
            tprint(f"Using {n_jobs} threads for processing (previous setting: {old_threads})")
        else:
            # Fall back to environment variable for older NumPy versions
            os.environ["OMP_NUM_THREADS"] = str(n_jobs)
            os.environ["MKL_NUM_THREADS"] = str(n_jobs)
            tprint(f"Set OMP_NUM_THREADS={n_jobs} for parallel processing")
    except Exception as e:
        tprint(f"Warning: Could not set NumPy thread count: {e}")
        tprint(f"Continuing with default thread settings")
    
    # NEW ORDER OF FILTERING STEPS:
    
    # STEP 1: Remove duplicate teams (moved from Step 4 to Step 1)
    if remove_duplicates:
        # Use a set for O(1) lookups instead of a list
        team_configs = set()
        unique_mask = np.ones(skill_matrix.shape[0], dtype=bool)
        
        # Process in batches to save memory
        batch_size = 10000  # Adjust this based on available memory
        num_batches = int(np.ceil(skill_matrix.shape[0] / batch_size))
        
        tprint(f"Step 1: Checking for duplicate teams in {num_batches} batches using {n_jobs} parallel jobs...")
        
        # Create a function to handle progress updates from worker processes
        def update_progress(processed, total):
            tprint(f"  Processed {processed}/{total} teams ({(processed/total)*100:.1f}%)")
        
        # Calculate number of batches per job to avoid too many small tasks
        n_jobs_adjusted = min(n_jobs, num_batches)
        
        # Process batches in parallel
        if n_jobs_adjusted > 1:
            # Break work into chunks for better load balancing
            results = []
            
            for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, skill_matrix.shape[0])
                
                # Process this batch
                local_mask, local_configs = process_batch(
                    batch_idx, start_idx, end_idx, skill_matrix, member_matrix, 
                    team_configs, update_progress
                )
                
                # Update global mask
                unique_mask[start_idx:end_idx] = local_mask
                
                # Update team_configs with new unique configs
                team_configs.update(local_configs)
        else:
            # Single-threaded version for small datasets or debugging
            for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, skill_matrix.shape[0])
                
                batch_skill_rows = skill_matrix[start_idx:end_idx]
                batch_member_rows = member_matrix[start_idx:end_idx]
                
                # Process each team in batch
                for i in range(end_idx - start_idx):
                    team_idx = start_idx + i
                    
                    # Get non-zero indices for this team
                    skill_row = batch_skill_rows[i]
                    member_row = batch_member_rows[i]
                    
                    skill_indices = tuple(skill_row.nonzero()[1])
                    member_indices = tuple(member_row.nonzero()[1])
                    
                    team_config = (skill_indices, member_indices)
                    
                    if team_config in team_configs:
                        unique_mask[team_idx] = False
                    else:
                        team_configs.add(team_config)
                    
                    # Print progress occasionally to avoid slowing down processing
                    if (team_idx + 1) % 100000 == 0:
                        tprint(f"  Processed {team_idx + 1}/{skill_matrix.shape[0]} teams ({((team_idx + 1)/skill_matrix.shape[0])*100:.1f}%)")
                
                # Clear memory after processing each batch
                del batch_skill_rows
                del batch_member_rows
        
        duplicate_count = np.sum(~unique_mask)
        filter_stats['duplicate_teams'] = int(duplicate_count)
        
        if duplicate_count > 0:
            tprint(f"Step 1: Removing {duplicate_count} duplicate teams")
            tprint("  Applying filters to matrices (this may take a few minutes for large datasets)...")
            
            # Track start time
            start_time = time.time()
            
            skill_matrix = skill_matrix[unique_mask]
            member_matrix = member_matrix[unique_mask]
            
            # Update cumulative mask
            where_true = np.where(cumulative_mask)[0]
            cumulative_mask[where_true] = unique_mask
            
            # Track end time and report
            end_time = time.time()
            elapsed = end_time - start_time
            tprint(f"  Filter application completed in {elapsed:.2f} seconds")
            
            # Recalculate both masks after filtering
            team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
            team_skill_sums = np.array(skill_matrix.sum(axis=1)).flatten()
            has_experts = team_expert_sums > 0
            has_skills = team_skill_sums > 0
            
            filter_stats['remaining_teams'] = skill_matrix.shape[0]
            tprint(f"Teams after removing duplicates: {skill_matrix.shape[0]}")
    
    # STEP 2: Remove teams with neither experts nor skills (previously Step 1)
    teams_with_neither_remain = np.sum(~has_experts & ~has_skills)
    if teams_with_neither_remain > 0:
        tprint(f"Step 2: Removing {teams_with_neither_remain} teams with neither experts nor skills")
        # Create a filter where teams have either experts or skills or both
        tprint("  Applying filters to matrices...")
        
        # Track start time
        start_time = time.time()
        
        neither_filter = has_experts | has_skills
        skill_matrix = skill_matrix[neither_filter]
        member_matrix = member_matrix[neither_filter]
        
        # Update cumulative mask
        where_true = np.where(cumulative_mask)[0]
        cumulative_mask[where_true] = neither_filter
        
        # Track end time and report
        end_time = time.time()
        elapsed = end_time - start_time
        tprint(f"  Filter application completed in {elapsed:.2f} seconds")
        
        # Recalculate both masks after filtering
        team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
        team_skill_sums = np.array(skill_matrix.sum(axis=1)).flatten()
        has_experts = team_expert_sums > 0
        has_skills = team_skill_sums > 0
        
        filter_stats['remaining_teams'] = skill_matrix.shape[0]
        tprint(f"Teams after removing those with neither experts nor skills: {skill_matrix.shape[0]}")
    
    # STEP 3: Remove teams with no experts (but have skills) (previously Step 2)
    if remove_zero_experts_teams:
        # Recalculate zero_expert_teams_only after previous steps
        zero_expert_teams_remain = np.sum(~has_experts & has_skills)
        if zero_expert_teams_remain > 0:
            filter_stats['zero_expert_teams_only'] = int(zero_expert_teams_remain)
            tprint(f"Step 3: Removing {zero_expert_teams_remain} teams with no experts (but have skills)")
            # Save the filter to apply to both matrices
            tprint("  Applying filters to matrices...")
            
            # Track start time
            start_time = time.time()
            
            experts_filter = has_experts
            skill_matrix = skill_matrix[experts_filter]
            member_matrix = member_matrix[experts_filter]
            
            # Update cumulative mask
            where_true = np.where(cumulative_mask)[0]
            cumulative_mask[where_true] = experts_filter
            
            # Track end time and report
            end_time = time.time()
            elapsed = end_time - start_time
            tprint(f"  Filter application completed in {elapsed:.2f} seconds")
            
            # Recalculate both masks after filtering
            team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
            team_skill_sums = np.array(skill_matrix.sum(axis=1)).flatten()
            has_experts = team_expert_sums > 0
            has_skills = team_skill_sums > 0
            
            filter_stats['remaining_teams'] = skill_matrix.shape[0]
            tprint(f"Teams after removing those with no experts: {skill_matrix.shape[0]}")
    
    # STEP 4: Remove teams with no skills (but have experts) (previously Step 3)
    if remove_zero_skills_teams:
        # Recalculate zero_skill_teams_only after previous steps
        zero_skill_teams_remain = np.sum(has_experts & ~has_skills) 
        if zero_skill_teams_remain > 0:
            filter_stats['zero_skill_teams_only'] = int(zero_skill_teams_remain)
            tprint(f"Step 4: Removing {zero_skill_teams_remain} teams with no skills (but have experts)")
            # Save the filter to apply to both matrices
            tprint("  Applying filters to matrices...")
            
            # Track start time
            start_time = time.time()
            
            skills_filter = has_skills
            skill_matrix = skill_matrix[skills_filter]
            member_matrix = member_matrix[skills_filter]
            
            # Update cumulative mask
            where_true = np.where(cumulative_mask)[0]
            cumulative_mask[where_true] = skills_filter
            
            # Track end time and report
            end_time = time.time()
            elapsed = end_time - start_time
            tprint(f"  Filter application completed in {elapsed:.2f} seconds")
            
            # Recalculate both masks after filtering
            team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
            team_skill_sums = np.array(skill_matrix.sum(axis=1)).flatten()
            has_experts = team_expert_sums > 0
            has_skills = team_skill_sums > 0
            
            filter_stats['remaining_teams'] = skill_matrix.shape[0]
            tprint(f"Teams after removing those with no skills: {skill_matrix.shape[0]}")
    
    # Handle minimum team and expert requirements if specified
    if minimum_teams is not None or team_size is not None:
        # Apply the minimum_teams and team_size filtering if provided
        if team_size is not None:
            tprint(f"\nSTEP 5: Filtering teams with < {team_size} experts")
            
        if minimum_teams is not None:
            tprint(f"\nSTEP 5: Filtering experts in < {minimum_teams} teams")
            
        # Initialize counters
        prev_valid_teams_count = 0
        prev_expert_count = 0
        iteration = 0
        max_iterations = 10  # Set a limit to avoid infinite loops
        
        # Track which teams pass all filters cumulatively
        cumulative_mask = np.ones(skill_matrix.shape[0], dtype=bool)
        
        # Iterate until convergence or max iterations reached
        while True:
            tprint(f"Filtering iteration {iteration + 1}")
            iteration += 1
            
            if iteration > max_iterations:
                tprint("Maximum iterations reached. Stopping filtration loop.")
                break
                
            iteration_mask = np.ones(skill_matrix.shape[0], dtype=bool)
            
            # Filter based on team size (min experts per team)
            if team_size is not None:
                tprint(f"  Checking teams with < {team_size} experts...")
                start_time = time.time()
                
                team_expert_sums = np.array(member_matrix.sum(axis=1)).flatten()
                iteration_mask = iteration_mask & (team_expert_sums >= team_size)
                
                end_time = time.time()
                tprint(f"  Check completed in {end_time - start_time:.2f} seconds")
                
                filter_stats['teams_below_min_experts'] += int(np.sum(~iteration_mask))
                current_valid_teams = np.sum(iteration_mask)
                tprint(f"  Valid teams after size filtering: {current_valid_teams}")
                
                if current_valid_teams == 0:
                    raise ValueError("All teams were filtered out!")
                    
                if current_valid_teams == prev_valid_teams_count:
                    break
                    
                skill_matrix = skill_matrix[iteration_mask]
                member_matrix = member_matrix[iteration_mask]
                prev_valid_teams_count = current_valid_teams

            # Now filter experts based on number of teams, if minimum_teams is provided
            if minimum_teams is not None:
                # Count how many teams each expert is in using the sparse matrix
                tprint(f"  Checking experts in < {minimum_teams} teams...")
                start_time = time.time()
                
                expert_team_counts = np.array(member_matrix.sum(axis=0)).flatten()
                experts_with_min_teams = expert_team_counts >= minimum_teams
                current_expert_count = np.sum(experts_with_min_teams)
                
                end_time = time.time()
                tprint(f"  Check completed in {end_time - start_time:.2f} seconds")
                
                tprint(f"  Experts with ≥{minimum_teams} teams: {current_expert_count}")
                
                if current_expert_count == 0:
                    raise ValueError("All experts were filtered out!")
                
                if current_expert_count == prev_expert_count:
                    # We've reached a stable state for experts
                    # Instead of removing teams, just zero out experts that don't meet minimum
                    # This requires converting to LIL format for efficient column operations
                    member_matrix_lil = member_matrix.tolil()
                    for col_idx in np.where(~experts_with_min_teams)[0]:
                        member_matrix_lil[:, col_idx] = 0
                    member_matrix = member_matrix_lil.tocsr()
                    break
                    
                # Zero out experts that don't meet minimum
                tprint("  Applying expert filter...")
                start_time = time.time()
                
                member_matrix_lil = member_matrix.tolil()
                for col_idx in np.where(~experts_with_min_teams)[0]:
                    member_matrix_lil[:, col_idx] = 0
                member_matrix = member_matrix_lil.tocsr()
                
                end_time = time.time()
                tprint(f"  Filter application completed in {end_time - start_time:.2f} seconds")
                
                prev_expert_count = current_expert_count
            else:
                # Skip expert filtering
                break

            # Remove teams that now have too few experts after zeroing out, if team_size is provided
            if team_size is not None:
                team_expert_counts = np.array(member_matrix.sum(axis=1)).flatten()
                teams_with_experts = team_expert_counts >= team_size
                
                teams_below_threshold = np.sum(~teams_with_experts)
                if teams_below_threshold > 0:
                    filter_stats['teams_below_min_experts'] += int(teams_below_threshold)
                    tprint(f"Removing {teams_below_threshold} teams with fewer than {team_size} experts")
                
                skill_matrix = skill_matrix[teams_with_experts]
                member_matrix = member_matrix[teams_with_experts]
                filter_stats['remaining_teams'] = skill_matrix.shape[0]

            # Apply the iteration mask to the cumulative mask
            where_true = np.where(cumulative_mask)[0]
            cumulative_mask[where_true] = iteration_mask
            
            if current_valid_teams == 0:
                raise ValueError("All teams were filtered out!")

    # Final statistics
    tprint("\nFinal statistics:")
    
    # Get expert participation counts
    final_expert_counts = np.array(member_matrix.sum(axis=0)).flatten()
    active_experts = final_expert_counts > 0
    
    if np.sum(active_experts) > 0:
        tprint(f"Min teams per expert: {np.min(final_expert_counts[active_experts])}")
        tprint(f"Max teams per expert: {np.max(final_expert_counts[active_experts])}")
    else:
        tprint("No active experts found!")
    
    tprint(f"\nFinal number of teams: {skill_matrix.shape[0]}")
    tprint(f"Final number of experts: {np.sum(active_experts)}")

    # Add verification step
    tprint("\nVerification after matrix updates:")
    tprint(f"skill_matrix shape: {skill_matrix.shape}")
    tprint(f"member_matrix shape: {member_matrix.shape}")

    # Update indexes
    if 't2i' in indexes:
        # Initialize cumulative_mask if it wasn't created in earlier filters
        if 'cumulative_mask' not in locals():
            cumulative_mask = np.ones(skill_matrix.shape[0], dtype=bool)
            
        kept_teams = np.where(cumulative_mask)[0]
        old_to_new = {old: new for new, old in enumerate(kept_teams)}
        
        # Create new t2i and i2t mappings
        new_t2i = {}
        new_i2t = {}
        
        for t, i in indexes['t2i'].items():
            if i in old_to_new:
                new_i = old_to_new[i]
                new_t2i[t] = new_i
                new_i2t[new_i] = t
        
        indexes['t2i'] = new_t2i
        indexes['i2t'] = new_i2t
        
        # Verify the mappings are one-to-one
        tprint("\nVerifying team index mappings:")
        tprint(f"Number of unique teams in t2i: {len(set(new_t2i.values()))}")
        tprint(f"Number of unique teams in i2t: {len(set(new_i2t.keys()))}")

    if 'm2i' in indexes:
        kept_members = np.where(active_experts)[0]
        old_to_new = {old: new for new, old in enumerate(kept_members)}
        indexes['m2i'] = {m: old_to_new[i] for m, i in indexes['m2i'].items() 
                         if i in old_to_new}
        indexes['i2m'] = {old_to_new[i]: m for m, i in indexes['i2m'].items() 
                         if i in old_to_new}

    # After all the filtering is done, before returning
    # Create an id vector for the remaining teams
    team_ids = np.arange(skill_matrix.shape[0])
    
    # Keep the data in sparse format
    fixed_teamsvecs = {
        'skill': skill_matrix,
        'member': member_matrix,
        'id': csr_matrix(team_ids.reshape(-1, 1))  # Add id vector
    }
    
    # At the end, update filter_stats with final count
    filter_stats['remaining_teams'] = skill_matrix.shape[0]
    
    # Log the full statistics
    tprint("\nFull filtering statistics:")
    tprint(f"Initial teams: {filter_stats['initial_teams']}")
    tprint(f"Teams with neither experts nor skills: {filter_stats['teams_with_neither']}")
    tprint(f"Teams with no experts (but have skills): {filter_stats['zero_expert_teams_only']}")
    tprint(f"Teams with no skills (but have experts): {filter_stats['zero_skill_teams_only']}")
    tprint(f"Duplicate teams: {filter_stats['duplicate_teams']}")
    tprint(f"Teams below minimum experts: {filter_stats['teams_below_min_experts']}")
    tprint(f"Final remaining teams: {filter_stats['remaining_teams']}")
    tprint(f"Total teams removed: {filter_stats['initial_teams'] - filter_stats['remaining_teams']}")
    
    # Calculate final statistics
    filter_stats['final_experts'] = member_matrix.shape[1]
    filter_stats['final_skills'] = skill_matrix.shape[1]
    
    return fixed_teamsvecs, indexes, filter_stats

def main():
    # Start the timer
    start_time = time.time()
    
    tprint("Starting data filter process...")
    
    parser = argparse.ArgumentParser(
        description="""
        Fix teams data by applying various filters based on the provided parameters.
        
        The script supports multiple filtering options that can be applied independently:
        - Remove duplicate teams (default: yes)
        - Remove teams with zero experts (default: yes)
        - Remove teams with zero skills (default: yes)
        - Filter experts based on minimum team participation (if provided)
        - Filter teams based on minimum number of experts (if provided)
        
        The filtered data will be saved to a new folder with a name that reflects the filtering parameters used.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # Disable built-in help to add it manually in a specific position
    )
    
    # First add the help option to its own group to appear at the top
    help_group = parser.add_argument_group('help')
    help_group.add_argument(
        '-h', '--help', 
        action='help', 
        default=argparse.SUPPRESS,
        help='Show this help message and exit'
    )
    
    # Add required arguments group - this will appear right after help
    required_group = parser.add_argument_group('required arguments')
    required_group.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='Path to the input DIRECTORY containing teamsvecs.pkl and indexes.pkl\n' + 
             'Example: data/preprocessed/gith/gith.data.csv.filtered'
    )
    
    required_group.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Dataset name to use in output (e.g., GITH, DBLP, IMDB, etc.)'
    )
    
    # Add optional arguments group
    optional_group = parser.add_argument_group('optional arguments')
    
    optional_group.add_argument(
        '-dr', '--duplicate-removal',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to remove duplicate teams (default: yes)'
    )
    
    optional_group.add_argument(
        '-mt', '--minimum-teams',
        type=int,
        help='Minimum number of teams an expert must be in\n' +
             'If not provided, this filter will not be applied'
    )
    
    optional_group.add_argument(
        '-ts', '--team-size',
        type=int,
        help='Minimum number of experts required per team\n' +
             'If not provided, this filter will not be applied'
    )
    
    optional_group.add_argument(
        '-zet', '--zero-experts-teams',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to remove teams with no experts (default: yes)'
    )
    
    optional_group.add_argument(
        '-zst', '--zero-skills-teams',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to remove teams with no skills (default: yes)'
    )
    
    optional_group.add_argument(
        '-r', '--run-reports',
        type=str,
        choices=['yes', 'no'],
        help='Whether to run data-reports.py on the filtered data after completion (no default)'
    )

    optional_group.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Custom output directory name (overrides automatic naming)'
    )

    optional_group.add_argument(
        '-t', '--threads',
        type=int,
        default=DEFAULT_THREADS,
        help=f'Number of parallel threads to use (default: {DEFAULT_THREADS})'
    )

    args = parser.parse_args()

    # Get the input directory
    input_path = Path(args.input_dir)
    
    # Use the provided dataset name
    dataset_name = args.dataset
    tprint(f"Using dataset name: {dataset_name}")
    
    # Construct paths
    teams_file = input_path / 'teamsvecs.pkl'
    indexes_file = input_path / 'indexes.pkl'
    teams_pkl_file = input_path / 'teams.pkl'
    
    try:
        with open(teams_file, 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError:
        tprint(f"Error: Could not find teamsvecs.pkl in {teams_file}")
        return
    
    try:
        # Load indexes file
        with open(indexes_file, 'rb') as f:
            indexes = pickle.load(f)
    except FileNotFoundError:
        tprint(f"Error: Could not find indexes.pkl in {indexes_file}")
        return
    
    # Check if teams.pkl exists
    has_teams_pkl = teams_pkl_file.exists()
    if has_teams_pkl:
        try:
            # First try to open the file to check if it's readable
            with open(teams_pkl_file, 'rb') as f:
                try:
                    teams_data = pickle.load(f)
                    tprint(f"Found and loaded teams.pkl from {teams_pkl_file}")
                except ModuleNotFoundError as e:
                    # Handle missing module errors (like 'cmn' not found)
                    tprint(f"Warning: teams.pkl requires a module that's not installed: {e}")
                    tprint("Skipping teams.pkl processing - the required module is missing")
                    has_teams_pkl = False
                    teams_data = None
                except Exception as e:
                    tprint(f"Warning: Found teams.pkl but couldn't properly load it: {e}")
                    has_teams_pkl = False
                    teams_data = None
        except Exception as e:
            tprint(f"Warning: Could not open teams.pkl file: {e}")
            has_teams_pkl = False
            teams_data = None
    else:
        tprint("Note: No teams.pkl found in source directory")
        teams_data = None
    
    # Parse filtering options
    remove_duplicates = args.duplicate_removal == 'yes'
    remove_zero_experts_teams = args.zero_experts_teams == 'yes'
    remove_zero_skills_teams = args.zero_skills_teams == 'yes'
    
    # Check if we're using just the default settings (all standard filters on, no min_teams or team_size)
    is_default_config = (remove_duplicates and 
                         remove_zero_experts_teams and 
                         remove_zero_skills_teams and
                         args.minimum_teams is None and
                         args.team_size is None)
    
    if is_default_config:
        # Use "unfiltered" for the baseline case with only default filters
        filter_string = "unfiltered"
    else:
        filter_parts = []
        
        # Only add boolean filters if different from default
        if not remove_duplicates:  # Default is True
            filter_parts.append(f"dup0")  # Use 0 for False
        
        # Add minimum teams if provided
        if args.minimum_teams is not None:
            filter_parts.append(f"mt{args.minimum_teams}")
        
        # Add team size if provided
        if args.team_size is not None:
            filter_parts.append(f"ts{args.team_size}")
        
        # Add zero experts teams filter if different from default
        if not remove_zero_experts_teams:  # Default is True
            filter_parts.append(f"zet0")  # Use 0 for False
        
        # Add zero skills teams filter if different from default
        if not remove_zero_skills_teams:  # Default is True
            filter_parts.append(f"zst0")  # Use 0 for False
        
        # Create the filter string
        filter_string = '_'.join(filter_parts)
    
    # Create the output folder name
    if args.output_dir:
        # Use the custom output directory name if provided
        output_folder_name = args.output_dir
    else:
        # Otherwise use the automatic naming convention
        output_folder_name = f"{input_path.name}_{filter_string}"
    
    output_path = input_path.parent / output_folder_name
    
    # Create the output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Construct output file paths
    output_file = output_path / 'teamsvecs.pkl'
    indexes_output_file = output_path / 'indexes.pkl'
    
    # Set threading parameters
    n_jobs = args.threads
    
    # Process teams
    try:
        fixed_teamsvecs, fixed_indexes, filter_stats = fix_data(
            teamsvecs, indexes,
            remove_duplicates=remove_duplicates,
            minimum_teams=args.minimum_teams, 
            team_size=args.team_size,
            remove_zero_experts_teams=remove_zero_experts_teams,
            remove_zero_skills_teams=remove_zero_skills_teams,
            n_jobs=n_jobs
        )
    except Exception as e:
        tprint(f"Error in fix_data: {e}")
        return
    
    # Save both files
    with open(output_file, 'wb') as f:
        pickle.dump(fixed_teamsvecs, f)
    with open(indexes_output_file, 'wb') as f:
        pickle.dump(fixed_indexes, f)
    
    # If teams.pkl was found in source, process and save it too
    if has_teams_pkl and teams_data is not None:
        # Get a list of team indices that were kept after filtering
        kept_team_indices = []
        if 't2i' in indexes and 't2i' in fixed_indexes:
            # For each team in the original data, check if it exists in the fixed data
            for team_id, old_idx in indexes['t2i'].items():
                if team_id in fixed_indexes['t2i']:
                    kept_team_indices.append(old_idx)
        
        # If we have team indices, filter the teams data
        if kept_team_indices:
            tprint(f"Filtering teams.pkl data to keep {len(kept_team_indices)} teams")
            filtered_teams_data = []
            
            # Sort indices for more efficient access
            kept_team_indices.sort()
            kept_idx_set = set(kept_team_indices)
            
            # Check the format of teams_data - it's usually a list of teams
            if isinstance(teams_data, list):
                # If it's indexed directly, just keep the teams at the kept indices
                if all(isinstance(idx, int) for idx in kept_team_indices) and max(kept_team_indices) < len(teams_data):
                    filtered_teams_data = [teams_data[idx] for idx in kept_team_indices]
                # Otherwise, try to match teams based on identifiers in case ordering differs
                else:
                    for team in teams_data:
                        # This requires teams to have identifiers that match keys in indexes
                        # The specific field to use depends on the data format
                        # Commonly it's 'id', 'name', or similar
                        if hasattr(team, 'id') and team.id in fixed_indexes['t2i']:
                            filtered_teams_data.append(team)
                        elif isinstance(team, dict) and 'id' in team and team['id'] in fixed_indexes['t2i']:
                            filtered_teams_data.append(team)
            
            # Save filtered teams data
            teams_output_file = output_path / 'teams.pkl'
            with open(teams_output_file, 'wb') as f:
                pickle.dump(filtered_teams_data, f)
            tprint(f"Saved filtered teams.pkl with {len(filtered_teams_data)} teams")
        else:
            tprint("Warning: Could not determine which teams to keep - teams.pkl not updated")
    
    # Save filter stats to a text file
    filter_stats_file = output_path / 'filter_stats.txt'
    with open(filter_stats_file, 'w') as f:
        f.write(f"Filter Statistics for {dataset_name} ({output_path.name})\n")
        f.write("=" * 40 + "\n")
        f.write(f"Initial teams: {filter_stats['initial_teams']:,}\n")
        
        # Add initial experts/skills counts if available
        if 'initial_experts' in filter_stats:
            f.write(f"Initial experts: {filter_stats['initial_experts']:,}\n")
        if 'initial_skills' in filter_stats:
            f.write(f"Initial skills: {filter_stats['initial_skills']:,}\n")
        
        f.write("\nRemoved:\n")
        f.write(f"- Duplicate teams: {filter_stats['duplicate_teams']:,}\n")
        f.write(f"- Zero expert AND skill teams: {filter_stats['teams_with_neither']:,}\n")
        f.write(f"- Zero expert teams (but have skills): {filter_stats['zero_expert_teams_only']:,}\n")
        f.write(f"- Zero skill teams (but have experts): {filter_stats['zero_skill_teams_only']:,}\n")
        
        if filter_stats['teams_below_min_experts'] > 0:
            f.write(f"- Teams below minimum experts: {filter_stats['teams_below_min_experts']:,}\n")
        
        total_removed = filter_stats['initial_teams'] - filter_stats['remaining_teams']
        f.write(f"\nTotal teams removed: {total_removed:,}\n\n")
        
        f.write(f"Final teams remaining: {filter_stats['remaining_teams']:,}\n")
        
        # Add final experts/skills counts if available and if filtering was done
        if 'final_experts' in filter_stats:
            f.write(f"Final experts remaining: {filter_stats['final_experts']:,}\n")
        if 'final_skills' in filter_stats:
            f.write(f"Final skills remaining: {filter_stats['final_skills']:,}\n")
    
    tprint(f"\nFixed data saved to new folder:")
    tprint(f"- {output_path}")
    tprint(f"- {output_file}")
    tprint(f"- {indexes_output_file}")
    if has_teams_pkl and 'teams_output_file' in locals():
        tprint(f"- {teams_output_file}")
    tprint(f"- {filter_stats_file}")
    tprint("\nRun data-reports.py on the filtered data to verify the changes.")
    
    # Check if run-reports is set to yes
    if args.run_reports == 'yes':
        tprint("\nRunning data-reports.py on the filtered data...")
        
        # Construct the command to run data-reports.py
        cmd = [
            'python', 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data-reports.py'),
            '--input-path', str(output_file),
            '--dataset', dataset_name,
            '--threads', str(n_jobs)  # Add thread count parameter
        ]
        
        # Run the command
        tprint(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    tprint(f"Data filtering completed successfully.")
    tprint(f"Total execution time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")

if __name__ == '__main__':
    main() 