#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cmn_v3 - New V3 Preprocessing Pipeline for Team-Based Datasets

This package provides classes and functions for preprocessing various team-based datasets:
- DBLP (academic publications)
- USPTO (patents)
- IMDB (movies)
- Gith (software repositories)

The preprocessing pipeline converts raw data into sparse vectors for machine learning models.
"""

from .team import Team, tprint
from .dblp import Publication
from .dblp_author import DblpAuthor

# .dblp_venue import removed as per requirements
from .gith import Repository
from .gith_contributor import GithContributor

__all__ = [
    "Team",
    "Publication",  # DBLP
    "DblpAuthor",  # DBLP-specific author class
    # "DblpVenue" removed as per requirements
    "Repository",  # Gith
    "GithContributor",  # Gith-specific contributor class
    "tprint",
]
