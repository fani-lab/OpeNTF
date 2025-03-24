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


def filter_ntop_skills(teamsvecs, indexes=None, ntop_skills=None, verbose=True):
    """
    Filter teams based on whether they contain skills not in the top N most common skills

    Args:
        teamsvecs: Dictionary containing sparse vectors for teams
        indexes: Dictionary containing indexes for skills and members (not used in this filter but kept for consistency)
        ntop_skills: Number of top skills to keep (None = no filtering)
        verbose: Whether to print progress information

    Returns:
        Filtered teamsvecs dictionary
    """
    if ntop_skills is None or ntop_skills <= 0:
        # No filtering needed
        if verbose:
            tprint("No top skills filtering applied (ntop_skills is None or <= 0)")
        return teamsvecs

    if verbose:
        tprint(f"Filtering teams with skills not in top {ntop_skills} skills...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get skill matrix
    skill_matrix = filtered_teamsvecs["skill"]

    # Initialize statistics
    original_teams = skill_matrix.shape[0]

    # Calculate skill frequencies (sum of each skill across all teams)
    skill_frequencies = np.array(skill_matrix.sum(axis=0)).flatten()

    # Get indices of top N skills by frequency
    top_skill_indices = np.argsort(skill_frequencies)[-ntop_skills:]

    # Create a mask for non-top skills
    non_top_skills_mask = np.ones(skill_matrix.shape[1], dtype=bool)
    non_top_skills_mask[top_skill_indices] = False

    # Find teams that have any non-top skills
    teams_with_non_top_skills = (
        np.array(skill_matrix[:, non_top_skills_mask].sum(axis=1)).flatten() > 0
    )

    # Find teams that meet the criteria (no non-top skills)
    valid_indices = np.where(~teams_with_non_top_skills)[0]

    # Count teams that don't meet the requirement
    removed_teams = original_teams - len(valid_indices)

    # If there are teams that don't meet the requirement, filter the matrices
    if removed_teams > 0:
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
    final_teams = original_teams - removed_teams
    removed_percentage = (
        (removed_teams / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {removed_teams:,} teams with skills not in top {ntop_skills} skills ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
