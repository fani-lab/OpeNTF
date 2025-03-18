"""
Filter function to remove experts that don't participate in enough teams.
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


def filter_min_teams_per_expert(
    teamsvecs, indexes=None, min_teams_per_expert=None, verbose=True
):
    """
    Filter experts who participate in too few teams

    Args:
        teamsvecs: Dictionary containing sparse vectors for teams
        indexes: Dictionary containing indexes for skills and members
        min_teams_per_expert: Minimum number of teams an expert must participate in (None = no filtering)
        verbose: Whether to print progress information

    Returns:
        Filtered teamsvecs dictionary
    """
    if min_teams_per_expert is None or min_teams_per_expert <= 0:
        # No filtering needed
        if verbose:
            tprint(
                "No minimum teams per expert filtering applied (min_teams_per_expert is None or <= 0)"
            )
        return teamsvecs

    if verbose:
        tprint(
            f"Filtering experts who participate in fewer than {min_teams_per_expert} teams..."
        )

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get member matrix
    member_matrix = filtered_teamsvecs["member"]

    # Initialize statistics
    original_teams = member_matrix.shape[0]
    original_experts = member_matrix.shape[1]

    # Calculate teams per expert (sum of teams for each expert)
    teams_per_expert = np.array(member_matrix.sum(axis=0)).flatten()

    # Find experts that meet the criteria
    valid_experts = np.where(teams_per_expert >= min_teams_per_expert)[0]

    # Count experts that don't meet the criteria
    removed_experts = original_experts - len(valid_experts)

    # If there are experts that don't meet the criteria, filter the matrices
    if removed_experts > 0:
        # Create a progress bar
        with tqdm(
            total=1,
            desc="Applying filter",
            ncols=80,
            bar_format="{l_bar}{bar:20}{r_bar}",
        ) as pbar:
            # Filter member matrix to keep only valid experts
            filtered_teamsvecs["member"] = member_matrix[:, valid_experts]
            pbar.update(1)

    # Calculate final statistics
    if verbose:
        tprint(
            f"Removed {removed_experts} experts who participate in fewer than {min_teams_per_expert} teams"
        )
        tprint(
            f"Retained {len(valid_experts)} experts out of {original_experts} ({len(valid_experts)/original_experts*100:.1f}%)"
        )

    return filtered_teamsvecs
