#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Timestamp printing utilities for consistent logging across the project.
"""

import pytz
from datetime import datetime


# Define EST timezone globally for reuse
EST = pytz.timezone('US/Eastern')

def get_est_time():
    """
    Get current time in EST timezone with formatted string
    
    Returns:
        str: Current time in format "YYYY-MM-DD HH:MM:SS TZ"
    """
    now = datetime.now(EST)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")

def tprint(message):
    """
    Print a message with timestamp in EST timezone
    
    Args:
        message (str): Message to print
    """
    timestamped_message = f"[{get_est_time()}] {message}"

    print(timestamped_message)

    return timestamped_message