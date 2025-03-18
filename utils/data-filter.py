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

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint, get_est_time

# Set default thread count to 75% of available cores, capped at 32
DEFAULT_THREADS = min(32, max(1, int(multiprocessing.cpu_count() * 0.75)))

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

def fix_data(teamsvecs, indexes, remove_dup_teams=True, minimum_teams=None, team_size=None,
            remove_empty_skills=False, remove_empty_members=False, 
            min_skills=None, max_skills=None):
    """Fix data issues and apply filters.
    
    Args:
        teamsvecs: Tuple of (teams_sparse, indexes)
        indexes: Dictionary of indexes
        remove_dup_teams: Whether to remove duplicate teams
        minimum_teams: Minimum number of teams an expert must be in
        team_size: Tuple of (min_size, max_size)
        remove_empty_skills: Whether to remove teams without skills
        remove_empty_members: Whether to remove teams without members
        min_skills: Minimum number of skills per team
        max_skills: Maximum number of skills per team
        
    Returns:
        Tuple of (filtered_teamsvecs, filtered_indexes, stats)
    """
    
    # Create statistics dictionary
    stats = {
        "Original teams": teamsvecs['skill'].shape[0],
        "Original members": indexes.get('member_id_map', {}).get('key_to_id', {}) if indexes else None,
        "Original skills": indexes.get('skill_id_map', {}).get('key_to_id', {}) if indexes else None,
        "Removed": []
    }
    
    # Get the number of original entities
    if stats["Original members"] is not None:
        stats["Original members"] = len(stats["Original members"])
    if stats["Original skills"] is not None:
        stats["Original skills"] = len(stats["Original skills"])
    
    tprint(f"Original data dimensions: {teamsvecs['skill'].shape}")
    
    # Step 1: Remove duplicate teams
    if remove_dup_teams:
        tprint("Removing duplicate teams...")
        
        n_teams = teamsvecs['skill'].shape[0]
        teams_before = n_teams
        
        # Dictionary to track seen configurations
        seen_configs = {}
        duplicate_indices = []
        
        # Detect duplicates
        for i in tqdm(range(n_teams), desc="Checking duplicates"):
            skill_row = teamsvecs['skill'][i]
            member_row = teamsvecs['member'][i]
            
            # Get the non-zero indices (skills and members for the team)
            skill_indices = tuple(skill_row.nonzero()[1])
            member_indices = tuple(member_row.nonzero()[1])
            
            # Create a configuration hash
            config = (skill_indices, member_indices)
            
            # Check if this configuration has been seen before
            if config in seen_configs:
                duplicate_indices.append(i)
            else:
                seen_configs[config] = i

        # Delete duplicates
        if duplicate_indices:
            tprint(f"Found {len(duplicate_indices)} duplicate teams")
            stats["Removed"].append(f"Duplicate teams: {len(duplicate_indices)}")
            
            # Keep track of the original indices that remain
            keep_indices = np.ones(n_teams, dtype=bool)
            keep_indices[duplicate_indices] = False
            remaining_indices = np.where(keep_indices)[0]
            
            # Create new sparse matrices without duplicates
            teamsvecs['skill'] = teamsvecs['skill'][remaining_indices]
            teamsvecs['member'] = teamsvecs['member'][remaining_indices]
            
            # Update all matrices in the teamsvecs dictionary
            for key in teamsvecs:
                if isinstance(teamsvecs[key], np.ndarray) and len(teamsvecs[key]) == teams_before:
                    teamsvecs[key] = teamsvecs[key][remaining_indices]
                elif hasattr(teamsvecs[key], 'shape') and teamsvecs[key].shape[0] == teams_before:
                    teamsvecs[key] = teamsvecs[key][remaining_indices]
            
            tprint(f"Removed {len(duplicate_indices)} duplicate teams")
            tprint(f"New data dimensions: {teamsvecs['skill'].shape}")
        else:
            tprint("No duplicate teams found")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine what to do
    domain = args.domain
    data_path = args.data_path
    output_path = args.output_path
    remove_dup_teams = args.duplicate_removal == 'yes'
    remove_empty_skills = args.empty_skills_removal == 'yes'
    remove_empty_members = args.empty_members_removal == 'yes'
    
    # Check if we're using default configuration
    is_default_config = (remove_dup_teams and
                        remove_empty_skills and 
                        remove_empty_members and
                        args.min_team_size == -1 and
                        args.max_team_size == -1 and
                        args.min_skills == -1 and
                        args.max_skills == -1 and
                        args.min_teams_per_expert == -1)
    
    # Add suffix for non-default configuration
    config_suffix = ""
    if not is_default_config:
        if not remove_dup_teams:  # Default is True
            config_suffix += "_keepdup"
        if not remove_empty_skills:  # Default is True
            config_suffix += "_keepemptyskills"
        if not remove_empty_members:  # Default is True
            config_suffix += "_keepemptymembers"
    
    # Call the fix_data function
    fixed_data, filtered_indexes, stats = fix_data(
        teamsvecs=teamsvecs,
        indexes=indexes,
        remove_dup_teams=remove_dup_teams,
        minimum_teams=min_teams_per_expert if min_teams_per_expert > 0 else None,
        team_size=(min_team_size, max_team_size) if min_team_size > 0 or max_team_size > 0 else None,
        remove_empty_skills=remove_empty_skills,
        remove_empty_members=remove_empty_members,
        min_skills=min_skills if min_skills > 0 else None,
        max_skills=max_skills if max_skills > 0 else None
    )

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
    remove_dup_teams = args.duplicate_removal == 'yes'
    remove_zero_experts_teams = args.zero_experts_teams == 'yes'
    remove_zero_skills_teams = args.zero_skills_teams == 'yes'
    
    # Check if we're using just the default settings (all standard filters on, no min_teams or team_size)
    is_default_config = (remove_dup_teams and 
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
        if not remove_dup_teams:  # Default is True
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
            remove_dup_teams=remove_dup_teams,
            minimum_teams=args.minimum_teams, 
            team_size=args.team_size,
            remove_empty_skills=remove_zero_skills_teams,
            remove_empty_members=remove_zero_experts_teams,
            min_skills=args.min_skills if args.min_skills > 0 else None,
            max_skills=args.max_skills if args.max_skills > 0 else None,
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