import pickle
import os
import numpy as np
from pathlib import Path

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
    
    # Second pass: assign indices
    for team_idx, (skill_row, member_row) in enumerate(zip(teams_skill_array, teams_member_array)):
        skill_indices = np.where(skill_row == 1)[0]
        expert_indices = np.where(member_row == 1)[0]
        team_config = (tuple(skill_indices), tuple(expert_indices))
        
        # If this config appears more than once, use the index, otherwise use '0'
        if seen_teams[team_config][1] > 1:
            dup_indices.append(str(seen_teams[team_config][0]))
            dup_teams += 1
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
        'dup_indices': dup_indices
    }

def main():
    # Get the directory of the current script
    script_dir = Path(__file__).parent.absolute()
    
    # Load teamsvecs.pkl from the same directory
    teams_file = script_dir / 'teamsvecs.pkl'
    
    try:
        with open(teams_file, 'rb') as f:
            teamsvecs = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find teamsvecs.pkl in {script_dir}")
        return
    
    # Analyze the data
    stats = analyze_teams(teamsvecs)
    
    # Create output file
    output_file = script_dir / 'teams_stat.csv'
    
    with open(output_file, 'w') as f:
        # Write summary stats
        f.write('stats,#teams,#skills,#experts,#zero_skill_teams,#zero_expert_teams,' +
                '#min_skills,#max_skills,#min_experts,#max_experts,' +
                '#min_exp_team,#max_exp_team,#dup_teams\n')
        f.write(f'summary,{stats["num_teams"]},{stats["num_skills"]},{stats["num_experts"]},' +
                f'{stats["zero_skill_teams"]},{stats["zero_expert_teams"]},' +
                f'{stats["min_skills"]},{stats["max_skills"]},' +
                f'{stats["min_experts"]},{stats["max_experts"]},' +
                f'{stats["min_exp_team"]},{stats["max_exp_team"]},' +
                f'{stats["dup_teams"]}\n\n')
        
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