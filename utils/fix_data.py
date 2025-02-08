import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

def fix_teams_data(teamsvecs):
    """
    Fix teams data by applying various filters:
    1. First remove experts that appear in less than 75 teams (but keep the teams)
    2. Remove teams with zero skills
    3. Remove teams with less than 3 experts
    4. Remove duplicate teams (keeping only one instance)
    """
    # Convert sparse matrices to dense for easier manipulation
    teams_skill_array = teamsvecs['skill'].toarray()
    teams_member_array = teamsvecs['member'].toarray()
    
    print(f"Initial number of teams: {len(teams_skill_array)}")
    print(f"Initial number of experts: {teams_member_array.shape[1]}")
    
    # First pass: Remove experts that don't meet minimum teams requirement
    expert_team_counts = np.sum(teams_member_array, axis=0)
    experts_with_min_teams = expert_team_counts >= 75
    
    print(f"Experts with ≥75 teams: {np.sum(experts_with_min_teams)}")
    
    # Keep only the columns (experts) that meet the minimum teams requirement
    teams_member_array = teams_member_array[:, experts_with_min_teams]
    
    # Now proceed with team filtering
    prev_valid_teams_count = 0
    iteration = 0
    
    while True:
        iteration += 1
        # Create masks for filtering
        has_skills = np.sum(teams_skill_array, axis=1) > 0
        experts_count = np.sum(teams_member_array, axis=1)
        has_min_experts = experts_count >= 3  # Teams must have at least 3 experts
        
        print(f"\nIteration {iteration}:")
        print(f"Teams with skills: {np.sum(has_skills)}")
        print(f"Teams with ≥3 experts: {np.sum(has_min_experts)}")
        
        # Combine initial masks
        valid_teams = has_skills & has_min_experts
        
        # Get unique team configurations
        team_configs = []
        unique_mask = np.ones(len(teams_skill_array), dtype=bool)
        
        for idx, (skill_row, member_row) in enumerate(zip(teams_skill_array, teams_member_array)):
            if not valid_teams[idx]:
                continue
                
            team_config = (tuple(np.where(skill_row == 1)[0]), 
                          tuple(np.where(member_row == 1)[0]))
            
            if team_config in team_configs:
                unique_mask[idx] = False
            else:
                team_configs.append(team_config)
        
        # Combine all filters
        final_mask = valid_teams & unique_mask
        
        # Check if we've converged (no more teams being filtered out)
        current_valid_teams = np.sum(final_mask)
        print(f"Valid teams after filtering: {current_valid_teams}")
        
        if current_valid_teams == 0:
            raise ValueError("All teams were filtered out! Please check the filtering criteria.")
            
        if current_valid_teams == prev_valid_teams_count:
            break
            
        # Update arrays for next iteration
        teams_skill_array = teams_skill_array[final_mask]
        teams_member_array = teams_member_array[final_mask]
        prev_valid_teams_count = current_valid_teams
    
    # Final statistics
    active_experts = np.sum(teams_member_array, axis=0) > 0
    print(f"\nFinal number of teams: {len(teams_skill_array)}")
    print(f"Final number of active experts: {np.sum(active_experts)}")
    
    # Convert back to sparse matrices
    fixed_teamsvecs = {
        'skill': csr_matrix(teams_skill_array),
        'member': csr_matrix(teams_member_array)
    }
    
    return fixed_teamsvecs

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 fix_data.py <dataset_name> <subfolder_name>")
        print("Example: python3 fix_data.py gith gith.data.csv.filtered.mt75.ts3")
        return

    dataset_name = sys.argv[1]
    subfolder_name = sys.argv[2]

    # Get the root project directory
    script_dir = Path(__file__).parent.parent.absolute()
    
    # Construct paths
    input_dir = script_dir.parent / 'data' / 'preprocessed' / dataset_name / subfolder_name
    teams_file = input_dir / 'teamsvecs.pkl'
    output_file = input_dir / 'teamsvecs_fixed.pkl'
    
    try:
        with open(teams_file, 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find teamsvecs.pkl in {teams_file}")
        return
    
    # Fix the data
    fixed_teamsvecs = fix_teams_data(teamsvecs)
    
    # Save the fixed data
    with open(output_file, 'wb') as f:
        pickle.dump(fixed_teamsvecs, f)
    
    print(f"Fixed data saved to {output_file}")
    print("Run team_stats.py on the fixed data to verify the changes.")

if __name__ == '__main__':
    main() 