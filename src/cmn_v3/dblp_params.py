#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from param import settings as COMMON_SETTINGS

"""
Parameters for DBLP dataset preprocessing.

This file contains domain-specific parameters for DBLP data processing.
"""

# Default parameters that can be overridden
DBLP_PARAMS = {
    # Domain-specific filters
    'pass1_filters': {
        # Common filters
        **COMMON_SETTINGS['data']['pass1_filters'],

        # Year range (inclusive)
        'min_year': -1, # -1, no filtering
        'max_year': -1,    # -1, no filtering
        
        'doc_types': None,    # List of document types to include (None = all)
    },
    'passn_filters': {
        **COMMON_SETTINGS['data']['passn_filters'],
    },
    'processing': {
        **COMMON_SETTINGS['data']['processing'],
    },
      # Keys to include in teamsvecs.pkl
    'output_keys': [
        'id', 
        'members',
        'skills',
        'year',
    ],
}
