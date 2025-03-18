"""
Filter function to remove duplicate teams from the dataset.

A duplicate team is defined as having the exact same combination of skills and members.
"""

import numpy as np
from utils.tprint import tprint
from tqdm import tqdm


def remove_dup_teams(teamsvecs, indexes=None, verbose=True):
    """
    Remove duplicate teams from the dataset.

    A duplicate team is defined as having the exact same combination of skills and members.

    Args:
        teamsvecs (dict): Dictionary containing team data with 'skill' and 'member' matrices
        indexes (dict): Dictionary of indexes (not used in this filter but kept for consistency)
        verbose (bool): Whether to print progress information

    Returns:
        dict: filtered_teamsvecs with duplicate teams removed
    """
    if verbose:
        tprint("Removing duplicate teams...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get the skill and member matrices
    skill_matrix = filtered_teamsvecs["skill"]
    member_matrix = filtered_teamsvecs["member"]

    # Initialize statistics
    original_teams = skill_matrix.shape[0]

    # Track unique team configurations
    seen_configs = {}
    unique_indices = []
    duplicate_indices = []

    # Identify unique and duplicate teams
    with tqdm(
        total=original_teams,
        desc="Finding duplicates",
        ncols=80,
        bar_format="{l_bar}{bar:20}{r_bar}",
    ) as pbar:
        for i in range(original_teams):
            skill_row = skill_matrix[i]
            member_row = member_matrix[i]

            # Get the non-zero indices (skills and members for the team)
            skill_indices = tuple(skill_row.nonzero()[1])
            member_indices = tuple(member_row.nonzero()[1])

            # Create a unique identifier for this team configuration
            team_config = (skill_indices, member_indices)

            if team_config in seen_configs:
                duplicate_indices.append(i)
            else:
                seen_configs[team_config] = i
                unique_indices.append(i)

            pbar.update(1)

    # Count duplicates
    duplicate_count = len(duplicate_indices)

    # If there are duplicates, filter the matrices
    if duplicate_count > 0:
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
                    filtered_teamsvecs[key] = filtered_teamsvecs[key][unique_indices]
                pbar.update(1)

    # Calculate final statistics
    final_teams = original_teams - duplicate_count
    removed_percentage = (
        (duplicate_count / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {duplicate_count:,} duplicate teams ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
