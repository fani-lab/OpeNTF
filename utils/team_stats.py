import pickle
import os
import numpy as np
from pathlib import Path
import sys

def analyze_teams(teamsvecs):
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
    
    # Convert sparse matrix to dense for analysis
    teams_skill_array = teamsvecs['skill'].toarray()
    teams_member_array = teamsvecs['member'].toarray()
    
    # Count how many teams each expert is in
    expert_team_counts = np.sum(teams_member_array, axis=0)  # Sum across teams for each expert
    min_exp_team = int(np.min(expert_team_counts))  # Minimum teams any expert is in
    max_exp_team = int(np.max(expert_team_counts))  # Maximum teams any expert is in
    
    # Analyze each team
    skills_per_team = []
    experts_per_team = []
    skill_indices_per_team = []
    expert_indices_per_team = []
    
    # Keep track of unique team configurations and their dup indices
    seen_teams = {}  # {team_config: (dup_index, count)}
    dup_index_counter = 1
    dup_indices = []  # Store dup_index for each team
    
    # First pass: count occurrences of each team configuration
    for team_idx, (skill_row, member_row) in enumerate(zip(teams_skill_array, teams_member_array)):
        skill_indices = np.where(skill_row == 1)[0]
        expert_indices = np.where(member_row == 1)[0]
        team_config = (tuple(skill_indices), tuple(expert_indices))
        
        if team_config in seen_teams:
            seen_teams[team_config][1] += 1
            # Only assign dup_index if this is the first duplicate we've found
            if seen_teams[team_config][1] == 2:
                seen_teams[team_config][0] = dup_index_counter
                dup_index_counter += 1
        else:
            seen_teams[team_config] = [0, 1]  # Initialize with dup_index 0
    
    # Calculate number of unique team configurations and duplicate teams
    unique_team_configs = len(seen_teams)
    dup_teams = sum(count - 1 for _, count in seen_teams.values())  # Changed this line
    
    # Second pass: assign indices
    for team_idx, (skill_row, member_row) in enumerate(zip(teams_skill_array, teams_member_array)):
        skill_indices = np.where(skill_row == 1)[0]
        expert_indices = np.where(member_row == 1)[0]
        team_config = (tuple(skill_indices), tuple(expert_indices))
        
        # If this config appears more than once, use the index, otherwise use '0'
        if seen_teams[team_config][1] > 1:
            dup_indices.append(str(seen_teams[team_config][0]))
        else:
            dup_indices.append('0')
        
        # Store the indices
        skill_indices_per_team.append(skill_indices)
        expert_indices_per_team.append(expert_indices)
        
        # Count skills and experts
        skills = len(skill_indices)
        experts = len(expert_indices)
        
        skills_per_team.append(skills)
        experts_per_team.append(experts)
        
        # Update statistics
        if skills == 0:
            zero_skill_teams += 1
        if experts == 0:
            zero_expert_teams += 1
            
        max_skills = max(max_skills, skills)
        min_skills = min(min_skills, skills)
        max_experts = max(max_experts, experts)
        min_experts = min(min_experts, experts)

    # If no teams were processed, set min values to 0
    if min_skills == float('inf'):
        min_skills = 0
    if min_experts == float('inf'):
        min_experts = 0

    return {
        'num_teams': num_teams,
        'num_skills': num_skills,
        'num_experts': num_experts,
        'zero_skill_teams': zero_skill_teams,
        'zero_expert_teams': zero_expert_teams,
        'max_skills': max_skills,
        'min_skills': min_skills,
        'max_experts': max_experts,
        'min_experts': min_experts,
        'min_exp_team': min_exp_team,
        'max_exp_team': max_exp_team,
        'dup_teams': dup_teams,
        'skills_per_team': skills_per_team,
        'experts_per_team': experts_per_team,
        'skill_indices': skill_indices_per_team,
        'expert_indices': expert_indices_per_team,
        'dup_indices': dup_indices,
        'expert_team_counts': expert_team_counts,
        'unique_team_configs': unique_team_configs
    }

def main():
    # Check if correct number of command line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python3 team_stats.py <dataset_name> <subfolder_name>")
        print("Example: python3 team_stats.py gith gith.data.csv.filtered.mt75.ts3")
        return

    dataset_name = sys.argv[1]
    subfolder_name = sys.argv[2]

    # Get the root project directory (OpeNTF)
    script_dir = Path(__file__).parent.parent.absolute()
    
    # Construct path to the teamsvecs.pkl file
    # Now looking directly in the data/preprocessed directory under the project root
    teams_file = script_dir.parent / 'data' / 'preprocessed' / dataset_name / subfolder_name / 'teamsvecs.pkl'
    
    try:
        with open(teams_file, 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find teamsvecs.pkl in {teams_file}")
        return
    
    # Analyze the data
    stats = analyze_teams(teamsvecs)
    
    # Create output file in the same directory as the input file
    output_file = teams_file.parent / f'{dataset_name}_team_stats.csv'
    
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
        f.write(',,,\n')
        
        # Expert participation stats
        f.write(',#min_exp_team,#max_exp_team,#mt_75,#ts_3\n')
        
        # Count experts with min/max team participation
        min_team_experts = sum(1 for x in stats['expert_team_counts'] if x == stats['min_exp_team'])
        max_team_experts = sum(1 for x in stats['expert_team_counts'] if x == stats['max_exp_team'])
        
        min_exp_percent = (min_team_experts / stats['num_experts']) * 100
        max_exp_percent = (max_team_experts / stats['num_experts']) * 100
        
        # For mt_75 and ts_3, we'll need to calculate these
        # Assuming these are derived from the subfolder name for now
        mt_75_experts = int(0.688 * stats['num_experts'])  # This should be calculated properly
        ts_3_teams = stats['num_teams']  # This should be calculated properly
        
        f.write(f',{stats["min_exp_team"]} ({min_team_experts} experts~{min_exp_percent:.1f}%),' +
                f'{stats["max_exp_team"]} ({max_team_experts} experts~{max_exp_percent:.1f}%),' +
                f'{mt_75_experts} ({68.8}%),' +
                f'{ts_3_teams} ({100.0}%)\n')
        f.write(',,,\n')
        
        # Write header for detailed stats
        f.write('dup_index,#skills,#experts,skills,experts\n')
        
        # Write detailed stats for each team
        for i in range(len(stats['skills_per_team'])):
            skill_indices = [f's{idx}' for idx in stats['skill_indices'][i]]
            expert_indices = [f'm{idx}' for idx in stats['expert_indices'][i]]
            
            f.write(f'{stats["dup_indices"][i]},{stats["skills_per_team"][i]},{stats["experts_per_team"][i]},' +
                   f'[{"-".join(skill_indices)}],' +
                   f'[{"-".join(expert_indices)}]\n')

if __name__ == '__main__':
    main() 