#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import List, Set, Dict, Any, Optional


def log_skills(
    output_path: str, teams: List[Any], skill_index: Optional[Dict] = None
) -> None:
    """
    Log all unique skills in the dataset, one per line

    Args:
        output_path: Path to store the logs
        teams: List of team objects with skills attribute
        skill_index: Optional dictionary mapping skill indices to skill names
    """
    if not output_path:
        return

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Define log file path
    skills_log_path = os.path.join(logs_dir, "skills.log")

    # Collect unique skills
    unique_skills = set()

    # If skill_index is provided, use it directly
    if skill_index:
        if isinstance(skill_index, dict):
            # Check if it's an i2s (index to skill) or s2i (skill to index) dictionary
            # Try to detect the format based on key types
            if all(isinstance(key, int) for key in list(skill_index.keys())[:10]):
                # Likely an i2s dictionary
                unique_skills = set(skill_index.values())
            else:
                # Likely an s2i dictionary
                unique_skills = set(skill_index.keys())
    # Otherwise extract from teams
    else:
        for team in teams:
            if hasattr(team, "skills") and team.skills:
                unique_skills.update(team.skills)

    # Sort skills alphabetically
    sorted_skills = sorted(unique_skills)

    # Initialize the log file
    with open(skills_log_path, "w", encoding="utf-8") as log_file:
        start_time = datetime.now()
        log_file.write(f"# Skills Log - Generated: {start_time}\n")
        log_file.write(f"# Total unique skills found: {len(sorted_skills)}\n")
        log_file.write(f"# Format: One skill per line\n")
        log_file.write(f"# {'='*80}\n\n")

        # Write one skill per line
        for skill in sorted_skills:
            log_file.write(f"{skill}\n")

    return sorted_skills


def log_skills_with_frequencies(
    output_path: str,
    teams: List[Any],
    skill_index: Optional[Dict] = None,
    member_skill_matrix=None,
) -> None:
    """
    Log all unique skills in the dataset with their frequencies

    Args:
        output_path: Path to store the logs
        teams: List of team objects with skills attribute
        skill_index: Optional dictionary mapping skill indices to skill names
        member_skill_matrix: Optional matrix of skill usage by members
    """
    if not output_path:
        return

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Define log file path
    skills_log_path = os.path.join(logs_dir, "skills_with_frequencies.log")

    # Collect unique skills and their frequencies
    skill_frequencies = {}

    # Extract skills and their frequencies
    if teams:
        for team in teams:
            if hasattr(team, "skills") and team.skills:
                for skill in team.skills:
                    if skill in skill_frequencies:
                        skill_frequencies[skill] += 1
                    else:
                        skill_frequencies[skill] = 1
    elif skill_index and member_skill_matrix:
        # Use the skill matrix to calculate frequencies if available
        import numpy as np

        if hasattr(member_skill_matrix, "toarray"):
            # Convert sparse matrix to dense for summing
            skill_sums = np.sum(member_skill_matrix.toarray(), axis=0)
        else:
            skill_sums = np.sum(member_skill_matrix, axis=0)

        # Map indices to skills and create frequency dictionary
        for idx, count in enumerate(skill_sums):
            if idx in skill_index:
                skill = skill_index[idx]
                skill_frequencies[skill] = int(count)

    # Sort skills by frequency (descending)
    sorted_skills = sorted(skill_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Initialize the log file
    with open(skills_log_path, "w", encoding="utf-8") as log_file:
        start_time = datetime.now()
        log_file.write(f"# Skills Frequency Log - Generated: {start_time}\n")
        log_file.write(f"# Total unique skills found: {len(sorted_skills)}\n")
        log_file.write(f"# Format: skill_name: frequency\n")
        log_file.write(f"# {'='*80}\n\n")

        # Write one skill per line with frequency
        for skill, frequency in sorted_skills:
            log_file.write(f"{skill}: {frequency}\n")

    return sorted_skills
