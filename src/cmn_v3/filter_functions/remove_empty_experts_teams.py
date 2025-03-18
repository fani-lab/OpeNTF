"""
Filter function to remove teams with no experts from the dataset.
"""

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


def remove_empty_experts_teams(teamsvecs, indexes=None, verbose=True):
    """
    Remove teams that have no experts/members

    Args:
        teamsvecs: Dictionary containing sparse vectors for teams
        indexes: Dictionary containing indexes for skills and members (not used in this filter but kept for consistency)
        verbose: Whether to print progress information

    Returns:
        Filtered teamsvecs dictionary
    """
    if verbose:
        tprint("Removing teams with no experts...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get member matrix
    member_matrix = filtered_teamsvecs["member"]

    # Initialize statistics
    original_teams = member_matrix.shape[0]

    # Calculate member counts (sum of members for each team)
    member_counts = np.array(member_matrix.sum(axis=1)).flatten()

    # Find teams with at least one member
    teams_with_experts = member_counts > 0

    # Count teams with no members
    empty_teams = np.sum(~teams_with_experts)

    # If there are teams with no members, filter the matrices
    if empty_teams > 0:
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
                    filtered_teamsvecs[key] = filtered_teamsvecs[key][
                        teams_with_experts
                    ]
                pbar.update(1)

    # Calculate final statistics
    final_teams = original_teams - empty_teams
    removed_percentage = (
        (empty_teams / original_teams * 100) if original_teams > 0 else 0
    )

    if verbose:
        tprint(
            f"Removed {empty_teams:,} teams with no experts ({removed_percentage:.2f}%)"
        )
        tprint(f"Teams after filtering: {final_teams:,}")

    return filtered_teamsvecs
