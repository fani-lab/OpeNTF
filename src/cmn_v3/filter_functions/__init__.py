#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter functions for preprocessing team data.

This package contains various filter functions that can be applied to team data
during preprocessing. Each filter function takes input data and returns a filtered
version of that data.

The filters are designed to be composable and can be applied in sequence.
"""

# Import all filter functions here for easy access
from .remove_dup_teams import remove_dup_teams
from .remove_empty_skills_teams import remove_empty_skills_teams
from .remove_empty_experts_teams import remove_empty_experts_teams
from .filter_min_team_size import filter_min_team_size
from .filter_max_team_size import filter_max_team_size
from .filter_min_skills_team import filter_min_skills_team
from .filter_max_skills_team import filter_max_skills_team
from .filter_min_teams_per_expert import filter_min_teams_per_expert
from .filter_max_teams_per_member import filter_max_teams_per_member

__all__ = [
    "remove_dup_teams",
    "remove_empty_skills_teams",
    "remove_empty_experts_teams",
    "filter_min_team_size",
    "filter_max_team_size",
    "filter_min_skills_team",
    "filter_max_skills_team",
    "filter_min_teams_per_expert",
    "filter_max_teams_per_member",
]
