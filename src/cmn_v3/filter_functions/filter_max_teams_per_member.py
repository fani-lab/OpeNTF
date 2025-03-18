#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint


def filter_max_teams_per_member(
    teamsvecs, indexes=None, max_teams_per_member=None, verbose=True
):
    """
    Filter members who participate in too many teams

    Args:
        teamsvecs: Dictionary containing sparse vectors for teams
        indexes: Dictionary containing indexes for skills and members (not used in this filter but kept for consistency)
        max_teams_per_member: Maximum number of teams a member can participate in (None = no filtering)
        verbose: Whether to print progress information

    Returns:
        Filtered teamsvecs dictionary
    """
    if max_teams_per_member is None or max_teams_per_member <= 0:
        # No filtering needed
        if verbose:
            tprint(
                "No maximum teams per member filtering applied (max_teams_per_member is None or <= 0)"
            )
        return teamsvecs

    if verbose:
        tprint(
            f"Filtering members who participate in more than {max_teams_per_member} teams..."
        )

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get member matrix
    member_matrix = filtered_teamsvecs["member"]

    # Initialize statistics
    original_teams = member_matrix.shape[0]
    original_members = member_matrix.shape[1]

    # Calculate teams per member (sum of teams for each member)
    teams_per_member = np.array(member_matrix.sum(axis=0)).flatten()

    # Find members that exceed the maximum
    excessive_members = np.where(teams_per_member > max_teams_per_member)[0]

    # If no members exceed the maximum, return the original data
    if len(excessive_members) == 0:
        if verbose:
            tprint(
                "No members exceed the maximum teams limit. Returning original data."
            )
        return filtered_teamsvecs

    # Create a progress bar for the filtering process
    with tqdm(
        total=len(excessive_members),
        desc="Processing members",
        ncols=80,
        bar_format="{l_bar}{bar:20}{r_bar}",
    ) as pbar:
        # Create a new member matrix to modify
        if hasattr(member_matrix, "tolil"):
            # Convert to lil_matrix for efficient row/column modifications
            filtered_member_matrix = member_matrix.tolil()
        else:
            # Already a dense array, convert to lil_matrix
            filtered_member_matrix = lil_matrix(member_matrix)

        # For each member that exceeds the maximum
        for member_idx in excessive_members:
            # Find teams this member is part of
            team_indices = np.where(
                member_matrix[:, member_idx].toarray().flatten() > 0
            )[0]

            # If the member is in more teams than allowed
            if len(team_indices) > max_teams_per_member:
                # Sort teams by some criteria (e.g., team size, skill count)
                # Here we'll just keep the first max_teams_per_member teams
                teams_to_keep = team_indices[:max_teams_per_member]
                teams_to_remove = team_indices[max_teams_per_member:]

                # Remove the member from the excess teams
                for team_idx in teams_to_remove:
                    filtered_member_matrix[team_idx, member_idx] = 0

            pbar.update(1)

        # Convert back to the original format
        if hasattr(member_matrix, "tocsr"):
            filtered_member_matrix = filtered_member_matrix.tocsr()

    # Find teams that still have at least one member
    team_member_counts = np.array(filtered_member_matrix.sum(axis=1)).flatten()
    valid_team_indices = np.where(team_member_counts > 0)[0]

    # Create the final filtered teamsvecs
    with tqdm(
        total=3, desc="Finalizing", ncols=80, bar_format="{l_bar}{bar:20}{r_bar}"
    ) as pbar:
        final_teamsvecs = {}

        # Filter ID vector
        final_teamsvecs["id"] = teamsvecs["id"][valid_team_indices]
        pbar.update(1)

        # Filter skill matrix
        final_teamsvecs["skill"] = teamsvecs["skill"][valid_team_indices]
        pbar.update(1)

        # Filter member matrix
        final_teamsvecs["member"] = filtered_member_matrix[valid_team_indices]
        pbar.update(1)

    # Calculate final statistics
    filtered_count = len(final_teamsvecs["id"])
    removed_count = original_teams - filtered_count

    if verbose:
        tprint(
            f"Processed {len(excessive_members)} members who exceed the maximum teams limit"
        )
        tprint(f"Removed {removed_count} teams that had no members after filtering")
        tprint(f"Teams after filtering: {filtered_count:,}")

    return final_teamsvecs
