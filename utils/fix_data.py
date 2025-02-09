import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import argparse

def fix_data(teamsvecs, indexes, remove_duplicates=True, minimum_teams=75, team_size=3):
    """
    Fix teams data and update indexes accordingly
    
    Args:
        teamsvecs: Dictionary containing skill and member sparse matrices
        indexes: Dictionary containing team and member indexes
        remove_duplicates: Whether to remove duplicate teams
        minimum_teams: Minimum number of teams an expert must be in (default: 75)
        team_size: Minimum number of experts per team (default: 3)
    """
    # Convert sparse matrices to dense for easier manipulation
    teams_skill_array = teamsvecs['skill'].toarray()
    teams_member_array = teamsvecs['member'].toarray()
    
    print(f"Initial number of teams: {len(teams_skill_array)}")
    print(f"Initial number of experts: {teams_member_array.shape[1]}")
    
    while True:  # Keep iterating until we have a stable set of experts and teams
        prev_expert_count = 0
        while True:  # Inner loop for expert filtering
            # First filter teams based on skills and experts count
            prev_valid_teams_count = 0
            cumulative_mask = np.ones(len(teams_skill_array), dtype=bool)
            iteration = 0
            
            while True:
                iteration += 1
                # Create masks for filtering
                has_skills = np.sum(teams_skill_array, axis=1) > 0
                experts_count = np.sum(teams_member_array, axis=1)
                has_min_experts = experts_count >= team_size  # Use team_size parameter
                
                print(f"\nIteration {iteration}:")
                print(f"Teams with skills: {np.sum(has_skills)}")
                print(f"Teams with ≥{team_size} experts: {np.sum(has_min_experts)}")
                
                valid_teams = has_skills & has_min_experts
                unique_mask = np.ones(len(teams_skill_array), dtype=bool)
                
                iteration_mask = valid_teams & unique_mask
                where_true = np.where(cumulative_mask)[0]
                cumulative_mask[where_true] = iteration_mask
                
                current_valid_teams = np.sum(iteration_mask)
                print(f"Valid teams after filtering: {current_valid_teams}")
                
                if current_valid_teams == 0:
                    raise ValueError("All teams were filtered out!")
                    
                if current_valid_teams == prev_valid_teams_count:
                    break
                    
                teams_skill_array = teams_skill_array[iteration_mask]
                teams_member_array = teams_member_array[iteration_mask]
                prev_valid_teams_count = current_valid_teams

            # Now filter experts based on number of teams
            expert_team_counts = np.sum(teams_member_array, axis=0)
            experts_with_min_teams = expert_team_counts >= minimum_teams  # Use minimum_teams parameter
            current_expert_count = np.sum(experts_with_min_teams)
            
            print(f"\nExperts with ≥{minimum_teams} teams: {current_expert_count}")
            
            if current_expert_count == 0:
                raise ValueError("All experts were filtered out!")
            
            if current_expert_count == prev_expert_count:
                # We've reached a stable state for experts
                # Instead of removing teams, just zero out experts that don't meet minimum
                teams_member_array[:, ~experts_with_min_teams] = 0
                break
                
            # Zero out experts that don't meet minimum
            teams_member_array[:, ~experts_with_min_teams] = 0
            prev_expert_count = current_expert_count

        # Remove teams that now have too few experts after zeroing out
        team_expert_counts = np.sum(teams_member_array, axis=1)
        teams_with_experts = team_expert_counts >= team_size  # Use team_size parameter
        
        teams_skill_array = teams_skill_array[teams_with_experts]
        teams_member_array = teams_member_array[teams_with_experts]
        
        # Handle duplicates if requested
        if remove_duplicates:
            team_configs = []
            unique_mask = np.ones(len(teams_skill_array), dtype=bool)
            
            for idx, (skill_row, member_row) in enumerate(zip(teams_skill_array, teams_member_array)):
                team_config = (tuple(np.where(skill_row == 1)[0]), 
                             tuple(np.where(member_row == 1)[0]))
                
                if team_config in team_configs:
                    unique_mask[idx] = False
                else:
                    team_configs.append(team_config)
            
            print(f"\nDuplicate teams found: {np.sum(~unique_mask)}")
            
            if np.sum(~unique_mask) > 0:  # If we found duplicates
                teams_skill_array = teams_skill_array[unique_mask]
                teams_member_array = teams_member_array[unique_mask]
                
                # Recheck expert counts after duplicate removal
                expert_team_counts = np.sum(teams_member_array, axis=0)
                if np.min(expert_team_counts[expert_team_counts > 0]) >= minimum_teams:
                    # All experts still have enough teams
                    break
                # If not all experts have enough teams, continue the outer loop
                continue
            else:
                # No duplicates found, we're done
                break
        else:
            # Not removing duplicates, we're done
            break

    # Final verification and cleanup
    final_expert_counts = np.sum(teams_member_array, axis=0)
    active_experts = final_expert_counts >= minimum_teams  # Use minimum_teams parameter
    
    # Remove completely inactive experts from the matrix
    teams_member_array = teams_member_array[:, active_experts]
    
    print("\nFinal expert statistics:")
    print(f"Min teams per expert: {np.min(final_expert_counts[active_experts])}")
    print(f"Max teams per expert: {np.max(final_expert_counts[active_experts])}")
    
    print(f"\nFinal number of teams: {len(teams_skill_array)}")
    print(f"Final number of experts: {np.sum(active_experts)}")

    # Add verification step
    print("\nVerification after matrix updates:")
    print(f"teams_skill_array shape: {teams_skill_array.shape}")
    print(f"teams_member_array shape: {teams_member_array.shape}")

    # Update indexes
    if 't2i' in indexes:
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
        print("\nVerifying team index mappings:")
        print(f"Number of unique teams in t2i: {len(set(new_t2i.values()))}")
        print(f"Number of unique teams in i2t: {len(set(new_i2t.keys()))}")

    if 'm2i' in indexes:
        kept_members = np.where(active_experts)[0]
        old_to_new = {old: new for new, old in enumerate(kept_members)}
        indexes['m2i'] = {m: old_to_new[i] for m, i in indexes['m2i'].items() 
                         if i in old_to_new}
        indexes['i2m'] = {old_to_new[i]: m for m, i in indexes['i2m'].items() 
                         if i in old_to_new}

    # After all the filtering is done, before returning
    # Create an id vector for the remaining teams
    team_ids = np.arange(teams_skill_array.shape[0])
    
    fixed_teamsvecs = {
        'skill': csr_matrix(teams_skill_array),
        'member': csr_matrix(teams_member_array),
        'id': csr_matrix(team_ids.reshape(-1, 1))  # Add id vector
    }
    
    return fixed_teamsvecs, indexes

def main():
    parser = argparse.ArgumentParser(
        description="""
        Fix teams data by applying various filters:
        1. Remove experts that appear in less than --minimum-teams teams (default: 75)
        2. Remove teams with zero skills
        3. Remove teams with less than --team-size experts (default: 3)
        4. Optionally remove duplicate teams
        5. Update corresponding indexes to maintain consistency
        
        Example usage:
        python fix_data.py gith gith.data.csv.filtered.mt75.ts3 --minimum-teams 50 --team-size 4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'dataset_name',
        type=str,
        help='Name of the dataset (e.g., gith, dblp)'
    )
    
    parser.add_argument(
        'subfolder_name',
        type=str,
        help='Name of the subfolder containing the data\n' +
             'Example: gith.data.csv.filtered.mt75.ts3\n' +
             'Note: The subfolder name can be different from the actual thresholds used'
    )
    
    parser.add_argument(
        '--remove-duplicates', '-rd',
        type=str,
        choices=['yes', 'no'],
        default='yes',
        help='Whether to remove duplicate teams (default: yes)'
    )

    parser.add_argument(
        '--minimum-teams', '-mt',
        type=int,
        default=75,
        help='Minimum number of teams an expert must participate in (default: 75)'
    )
    
    parser.add_argument(
        '--team-size', '-ts',
        type=int,
        default=3,
        help='Minimum number of experts required per team (default: 3)'
    )

    args = parser.parse_args()

    # Get the root project directory
    script_dir = Path(__file__).parent.parent.absolute()
    
    # Construct paths
    input_dir = script_dir.parent / 'data' / 'preprocessed' / args.dataset_name / args.subfolder_name
    teams_file = input_dir / 'teamsvecs.pkl'
    output_file = input_dir / 'teamsvecs_fixed.pkl'
    
    try:
        with open(teams_file, 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find teamsvecs.pkl in {teams_file}")
        return
    
    try:
        # Load indexes file
        with open(input_dir / 'indexes.pkl', 'rb') as f:
            indexes = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find indexes.pkl in {input_dir}")
        return
    
    # Fix both data structures
    remove_duplicates = args.remove_duplicates.lower() == 'yes'
    fixed_teamsvecs, fixed_indexes = fix_data(
        teamsvecs, 
        indexes, 
        remove_duplicates=remove_duplicates,
        minimum_teams=args.minimum_teams,
        team_size=args.team_size
    )
    
    # Save both files
    with open(output_file, 'wb') as f:
        pickle.dump(fixed_teamsvecs, f)
    with open(input_dir / 'indexes_fixed.pkl', 'wb') as f:
        pickle.dump(fixed_indexes, f)
    
    print(f"\nFixed data saved to:")
    print(f"- {output_file}")
    print(f"- {input_dir / 'indexes_fixed.pkl'}")
    print("\nRun team_stats.py on the fixed data to verify the changes.")

if __name__ == '__main__':
    main() 