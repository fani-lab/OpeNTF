import pickle
import os
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import pytz
import time

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint, get_est_time

# Import the modular filter functions using absolute imports instead of relative
# This prevents the ImportError when running as a script
from src.cmn_v3.filter_functions.remove_dup_teams import remove_dup_teams
from src.cmn_v3.filter_functions.remove_empty_skills_teams import (
    remove_empty_skills_teams,
)
from src.cmn_v3.filter_functions.remove_empty_experts_teams import (
    remove_empty_experts_teams,
)
from src.cmn_v3.filter_functions.filter_min_team_size import filter_min_team_size
from src.cmn_v3.filter_functions.filter_max_team_size import filter_max_team_size
from src.cmn_v3.filter_functions.filter_min_skills_team import filter_min_skills_team
from src.cmn_v3.filter_functions.filter_max_skills_team import filter_max_skills_team
from src.cmn_v3.filter_functions.filter_min_teams_per_expert import (
    filter_min_teams_per_expert,
)
from src.cmn_v3.filter_functions.filter_max_teams_per_member import (
    filter_max_teams_per_member,
)

# Set default number of threads based on CPU cores
DEFAULT_THREADS = min(
    max(multiprocessing.cpu_count() - 2, 1), 16
)  # Use at most 16 threads by default


def apply_filters(teamsvecs, indexes=None, domain_params=None):
    """
    Apply filters to teams data based on domain parameters.
    This is a wrapper around the modular filter functions from filter_functions.

    Args:
        teamsvecs: Dictionary containing team data
        indexes: Dictionary of indexes (optional)
        domain_params: Dictionary of domain-specific parameters (optional)

    Returns:
        dict: filtered_teamsvecs
    """
    tprint("Applying filters using modular filter functions...")

    # Create a copy of the input to avoid modifying the original
    filtered_teamsvecs = {
        k: v.copy() if hasattr(v, "copy") else v for k, v in teamsvecs.items()
    }

    # Get initial count of teams
    original_teams = filtered_teamsvecs["skill"].shape[0]
    tprint(f"Initial team count: {original_teams:,}")

    # If no domain_params provided, return original data
    if not domain_params or "filters" not in domain_params:
        tprint("No domain parameters provided. Skipping filters.")
        return filtered_teamsvecs

    filters = domain_params.get("filters", {})
    common_filters = filters.get("common", {})

    # Get the domain name from domain_params if available
    domain_name = domain_params.get("domain_name", "")

    # Get domain-specific filters using the domain name
    domain_filters = filters.get(domain_name, {}) if domain_name else {}

    # If domain name is not available, try to find it from the filters keys
    if not domain_filters:
        # Look for domain keys (gith, dblp, etc.) in filters
        domain_keys = [k for k in filters.keys() if k != "common" and k != "domain"]
        if domain_keys:
            # Use the first available domain key
            domain_name = domain_keys[0]
            domain_filters = filters.get(domain_name, {})
            tprint(f"Using domain filters for: {domain_name}")

    # Track remaining teams for logging
    remaining = original_teams

    # Apply filters in a specific order for best performance

    # 1. First remove duplicate teams if requested
    if common_filters.get("remove_dup_teams", False):
        tprint("Removing duplicate teams...")
        filtered_teamsvecs = remove_dup_teams(filtered_teamsvecs, indexes)
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after removing duplicates: {remaining:,}")

    # 2. Remove teams with no skills/experts
    if common_filters.get("remove_empty_skills_teams", False):
        tprint("Removing teams with no skills...")
        filtered_teamsvecs = remove_empty_skills_teams(filtered_teamsvecs, indexes)
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after removing empty skills teams: {remaining:,}")

    if common_filters.get("remove_empty_experts_teams", False):
        tprint("Removing teams with no experts...")
        filtered_teamsvecs = remove_empty_experts_teams(filtered_teamsvecs, indexes)
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after removing empty experts teams: {remaining:,}")

    # 3. Apply team size filters
    min_team_size = common_filters.get("min_team_size")
    if min_team_size is not None and min_team_size > 0:
        tprint(f"Filtering teams with fewer than {min_team_size} experts...")
        filtered_teamsvecs = filter_min_team_size(
            filtered_teamsvecs, indexes, min_team_size
        )
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after min team size filter: {remaining:,}")

    max_team_size = common_filters.get("max_team_size")
    if max_team_size is not None and max_team_size > 0:
        tprint(f"Filtering teams with more than {max_team_size} experts...")
        filtered_teamsvecs = filter_max_team_size(
            filtered_teamsvecs, indexes, max_team_size
        )
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after max team size filter: {remaining:,}")

    # 4. Apply skills filters
    min_skills = common_filters.get("min_skills")
    if min_skills is not None and min_skills > 0:
        tprint(f"Filtering teams with fewer than {min_skills} skills...")
        filtered_teamsvecs = filter_min_skills_team(
            filtered_teamsvecs, indexes, min_skills
        )
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after min skills filter: {remaining:,}")

    max_skills = common_filters.get("max_skills")
    if max_skills is not None and max_skills > 0:
        tprint(f"Filtering teams with more than {max_skills} skills...")
        filtered_teamsvecs = filter_max_skills_team(
            filtered_teamsvecs, indexes, max_skills
        )
        remaining = filtered_teamsvecs["skill"].shape[0]
        tprint(f"Remaining teams after max skills filter: {remaining:,}")

    # 5. Apply minimum teams per expert filter
    min_teams_per_expert = common_filters.get("min_teams_per_expert")
    if min_teams_per_expert is not None and min_teams_per_expert > 0:
        tprint(f"Filtering experts with fewer than {min_teams_per_expert} teams...")
        filtered_teamsvecs = filter_min_teams_per_expert(
            filtered_teamsvecs, indexes, min_teams_per_expert
        )
        # This filter may not change team count, so no need to update 'remaining'

    # 6. Apply maximum teams per member filter
    max_teams_per_member = common_filters.get("max_teams_per_member")
    if max_teams_per_member is not None and max_teams_per_member > 0:
        tprint(f"Filtering experts with more than {max_teams_per_member} teams...")
        filtered_teamsvecs = filter_max_teams_per_member(
            filtered_teamsvecs, indexes, max_teams_per_member
        )
        # This filter may not change team count, so no need to update 'remaining'

    # Calculate final statistics
    final_teams = filtered_teamsvecs["skill"].shape[0]
    removed_teams = original_teams - final_teams
    removal_percentage = (
        (removed_teams / original_teams * 100) if original_teams > 0 else 0
    )

    tprint(f"Filter application complete:")
    tprint(f"- Original teams: {original_teams:,}")
    tprint(f"- Final teams: {final_teams:,}")
    tprint(f"- Removed teams: {removed_teams:,} ({removal_percentage:.2f}%)")

    return filtered_teamsvecs
