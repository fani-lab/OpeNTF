#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import sys
from pathlib import Path
import torch
from utils.tprint import tprint
from tqdm import tqdm

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def filter_max_team_size(teamsvecs, indexes=None, max_size=None, verbose=True):
    """
    Remove teams that have more experts than the specified maximum team size.

    Args:
        teamsvecs (dict): Dictionary containing team data with 'skill' and 'member' matrices
        indexes (dict): Dictionary of indexes (not used in this filter but kept for consistency)
        max_size (int): Maximum number of experts allowed per team
        verbose (bool): Whether to print progress information

    Returns:
        dict: filtered_teamsvecs with teams that meet the maximum size requirement
    """
    if max_size is None or max_size <= 0:
        if verbose:
            tprint("No maximum team size filtering applied (max_size is None or <= 0)")
        return teamsvecs

    if verbose:
        tprint(f"Filtering teams with more than {max_size} experts...")

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

    # Identify teams that meet the maximum size requirement
    valid_indices = np.where(expert_counts <= max_size)[0]

    # Count teams that don't meet the requirement
    large_teams = original_teams - len(valid_indices)

    # If there are teams that don't meet the requirement, filter the matrices
    if large_teams > 0:
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
    final_teams = original_teams - large_teams
    removed_percentage = (
        (large_teams / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {large_teams:,} teams with more than {max_size} experts ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
