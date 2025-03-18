import os
from datetime import datetime
from utils.tprint import tprint


def create_toy_logs_folder(toy_output_path, toy_teams):
    """
    Create a logs folder for the toy dataset with entries_processed.log and skills.log

    Args:
        toy_output_path: Path to the toy dataset directory
        toy_teams: List of team objects in the toy dataset
    """
    logs_dir = os.path.join(toy_output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create entries_processed.log
    entries_processed_path = os.path.join(logs_dir, "entries_processed.log")
    with open(entries_processed_path, "w", encoding="utf-8") as f:
        f.write(f"Total entries processed: {len(toy_teams)}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Add some sample entries
        for i, team in enumerate(toy_teams[:10]):  # Show first 10 teams
            team_id = getattr(team, "id", f"team_{i}")
            team_name = getattr(team, "name", getattr(team, "title", team_id))
            num_members = len(getattr(team, "members", []))
            num_skills = len(getattr(team, "skills", []))
            f.write(
                f"Processed: {team_name} (ID: {team_id}) - {num_members} members, {num_skills} skills\n"
            )

        if len(toy_teams) > 10:
            f.write(f"... and {len(toy_teams) - 10} more entries\n")

    # Create skills.log
    skills_path = os.path.join(logs_dir, "skills.log")
    with open(skills_path, "w", encoding="utf-8") as f:
        f.write(f"Skills from toy dataset with {len(toy_teams)} teams\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Collect and count all skills
        all_skills = {}
        for team in toy_teams:
            for skill in getattr(team, "skills", []):
                if skill in all_skills:
                    all_skills[skill] += 1
                else:
                    all_skills[skill] = 1

        # Sort skills by frequency
        sorted_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)

        # Write skills to file
        for skill, count in sorted_skills:
            f.write(f"{skill}: {count} occurrences\n")

    tprint(f"Created logs folder for toy dataset:")
    tprint(f"  - {entries_processed_path}")
    tprint(f"  - {skills_path}")
