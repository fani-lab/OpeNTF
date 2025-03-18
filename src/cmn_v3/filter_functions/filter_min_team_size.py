"""
Filter function to remove teams that don't meet the minimum team size requirement.
"""

import numpy as np
from utils.tprint import tprint
from tqdm import tqdm


def filter_min_team_size(teamsvecs, indexes=None, min_size=2, verbose=True):
    """
    Remove teams that have fewer experts than the specified minimum team size.

    Args:
        teamsvecs (dict): Dictionary containing team data with 'skill' and 'member' matrices
        indexes (dict): Dictionary of indexes (not used in this filter but kept for consistency)
        min_size (int): Minimum number of experts required per team
        verbose (bool): Whether to print progress information

    Returns:
        dict: filtered_teamsvecs with teams that meet the minimum size requirement
    """
    if min_size is None or min_size <= 0:
        if verbose:
            tprint("No minimum team size filtering applied (min_size is None or <= 0)")
        return teamsvecs

    if verbose:
        tprint(f"Filtering teams with fewer than {min_size} experts...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get the member matrix
    member_matrix = filtered_teamsvecs["member"]

    # Initialize statistics
    original_teams = member_matrix.shape[0]

    # Calculate the number of experts per team
    expert_counts = np.array(member_matrix.sum(axis=1)).flatten()

    # Identify teams that meet the minimum size requirement
    valid_teams = expert_counts >= min_size

    # Count teams that don't meet the requirement
    small_teams = np.sum(~valid_teams)

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
                    filtered_teamsvecs[key] = filtered_teamsvecs[key][valid_teams]
                pbar.update(1)

    # Calculate final statistics
    final_teams = original_teams - small_teams
    removed_percentage = (
        (small_teams / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {small_teams:,} teams with fewer than {min_size} experts ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
