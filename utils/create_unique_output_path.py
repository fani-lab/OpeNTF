import os
from datetime import datetime


def create_unique_output_path(base_path):
    """
    Create a unique output path by appending a timestamp if the base path already exists.

    Args:
        base_path: Base directory path

    Returns:
        Unique path (original path if it doesn't exist, otherwise with timestamp appended)
    """
    if not os.path.exists(base_path):
        return base_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_path}_{timestamp}"
