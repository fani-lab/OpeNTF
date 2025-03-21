import os
from datetime import datetime


def create_unique_output_path(base_path):
    """
    Create a unique output path by appending a numeric suffix if the base path already exists.

    Args:
        base_path: Base directory path

    Returns:
        Unique path (original path if it doesn't exist, otherwise with numeric suffix)
    """
    if not os.path.exists(base_path):
        return base_path

    # Use numeric suffixes instead of timestamps
    counter = 1
    while True:
        new_path = f"{base_path}_{counter}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1
