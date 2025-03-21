from utils.tprint import tprint

def parse_gpus_string(gpus_str):
    """
    Parse a GPU string specification into a list of device indices

    Args:
        gpus_str: String specification of GPU devices (e.g., "0", "0,1,2")

    Returns:
        List of integers representing GPU device indices, or special strings like "all" or "first"
    """
    if not gpus_str:
        return None

    if isinstance(gpus_str, list):
        # Already a list of indices
        return gpus_str

    if not isinstance(gpus_str, str):
        return None

    # Handle special strings
    if gpus_str.lower() == "all":
        return "all"
    if gpus_str.lower() == "first":
        return "first"

    # Parse comma-separated indices
    try:
        if "," in gpus_str:
            return [int(idx.strip()) for idx in gpus_str.split(",")]
        else:
            return [int(gpus_str.strip())]
    except ValueError:
        tprint(f"Invalid GPU specification: {gpus_str}. Using first available GPU.")
        return "first"
