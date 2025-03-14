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
Parameters for Gith dataset preprocessing.

This file contains domain-specific parameters for Gith data processing.
"""

# Default parameters that can be overridden
GITH_PARAMS = {
    # Domain-specific filters
    'pass1_filters': {
        # Common filters - Override with more relaxed settings
        **COMMON_SETTINGS['data']['pass1_filters'],
        "remove_overloaded_repos": False,

        # Year range (inclusive)
        'min_year': -1,  # Accept repositories from 2000 onwards
        'max_year': -1,  # Accept repositories up to 2025
        
        # Repository specific filters - Very relaxed
        'languages': None, # Accept all languages
    },
    'passn_filters': {
        **COMMON_SETTINGS['data']['passn_filters'],
    },
    'processing': {
        **COMMON_SETTINGS['data']['processing'],
        'debug_logs': True,  # Enable debug logs to help diagnose issues
    },
    # Keys to include in teamsvecs.pkl
    'output_keys': [
        'id', 
        'members',
        'skills',
        'year',
    ]
} 