#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint


def filter_min_skills_team(teamsvecs, indexes=None, min_skills_team=None, verbose=True):
    """
    Filter teams based on minimum number of skills per team

    Args:
        teamsvecs: Dictionary containing sparse vectors for teams
        indexes: Dictionary containing indexes for skills and members (not used in this filter but kept for consistency)
        min_skills_team: Minimum number of skills a team must have (None = no filtering)
        verbose: Whether to print progress information

    Returns:
        Filtered teamsvecs dictionary
    """
    if min_skills_team is None or min_skills_team <= 0:
        # No filtering needed
        if verbose:
            tprint(
                "No minimum skills filtering applied (min_skills_team is None or <= 0)"
            )
        return teamsvecs

    if verbose:
        tprint(f"Filtering teams with fewer than {min_skills_team} skills...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get skill matrix
    skill_matrix = filtered_teamsvecs["skill"]

    # Initialize statistics
    original_teams = skill_matrix.shape[0]

    # Calculate skill counts (sum of skills for each team)
    skill_counts = np.array(skill_matrix.sum(axis=1)).flatten()

    # Find teams that meet the criteria
    valid_indices = np.where(skill_counts >= min_skills_team)[0]

    # Count teams that don't meet the requirement
    small_teams = original_teams - len(valid_indices)

    # If there are teams that don't meet the requirement, filter the matrices
    if small_teams > 0:
        # Create a progress bar
        with tqdm(
            total=len(filtered_teamsvecs),
            desc="Applying filter",
            ncols=80,
            bar_format="{l_bar}{bar:20}{r_bar}",
        ) as pbar:
            for key in filtered_teamsvecs:
                if (
                    hasattr(filtered_teamsvecs[key], "shape")
                    and filtered_teamsvecs[key].shape[0] == original_teams
                ):
                    filtered_teamsvecs[key] = filtered_teamsvecs[key][valid_indices]
                pbar.update(1)

    # Calculate final statistics
    final_teams = original_teams - small_teams
    removed_percentage = (
        (small_teams / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {small_teams:,} teams with fewer than {min_skills_team} skills ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
