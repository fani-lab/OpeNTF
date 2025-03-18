#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to import GPU libraries and check their availability.
"""

from utils.tprint import tprint

# Globals that will be used by other modules
CUPY_AVAILABLE = False
SELECTED_GPU_DEVICES = None


def parse_gpu_string(gpu_str):
    """
    Parse a GPU string specification into a list of device indices

    Args:
        gpu_str: String specification of GPU devices (e.g., "0", "0,1,2")

    Returns:
        List of integers representing GPU device indices, or special strings like "all" or "first"
    """
    if not gpu_str:
        return None

    if isinstance(gpu_str, list):
        # Already a list of indices
        return gpu_str

    if not isinstance(gpu_str, str):
        return None

    # Handle special strings
    if gpu_str.lower() == "all":
        return "all"
    if gpu_str.lower() == "first":
        return "first"

    # Parse comma-separated indices
    try:
        if "," in gpu_str:
            return [int(idx.strip()) for idx in gpu_str.split(",")]
        else:
            return [int(gpu_str.strip())]
    except ValueError:
        tprint(f"Invalid GPU specification: {gpu_str}. Using first available GPU.")
        return "first"


def import_gpu_libs():
    """
    Try to import GPU acceleration libraries and return availability status

    Returns:
        bool: True if GPU libraries are available, False otherwise
    """
    global CUPY_AVAILABLE, SELECTED_GPU_DEVICES
    try:
        import cupy as cp

        CUPY_AVAILABLE = True

        # Get GPU device information if available
        try:
            device_count = cp.cuda.runtime.getDeviceCount()

            # Parse SELECTED_GPU_DEVICES if it's a string
            if isinstance(SELECTED_GPU_DEVICES, str):
                SELECTED_GPU_DEVICES = parse_gpu_string(SELECTED_GPU_DEVICES)

            # Handle the special modes for GPU selection
            if SELECTED_GPU_DEVICES == "all":
                # Use all available GPUs
                SELECTED_GPU_DEVICES = list(range(device_count))
                tprint(f"Using all {device_count} available GPU devices")

            elif SELECTED_GPU_DEVICES == "first":
                # Use only the first GPU
                SELECTED_GPU_DEVICES = [0]
                tprint(f"Using only the first GPU device")

            elif isinstance(SELECTED_GPU_DEVICES, list) and all(
                isinstance(x, int) for x in SELECTED_GPU_DEVICES
            ):
                # Validate that all specified devices are available
                valid_devices = [d for d in SELECTED_GPU_DEVICES if d < device_count]
                if len(valid_devices) != len(SELECTED_GPU_DEVICES):
                    tprint(
                        f"Warning: Some specified GPU devices are not available. Using only: {valid_devices}"
                    )
                SELECTED_GPU_DEVICES = valid_devices

            else:
                # Default to using all GPUs if the selection is invalid
                SELECTED_GPU_DEVICES = list(range(device_count))
                tprint(f"Using all {device_count} available GPU devices (default)")

            # Print device information
            for device_id in SELECTED_GPU_DEVICES:
                cp.cuda.Device(device_id).use()
                device_props = cp.cuda.runtime.getDeviceProperties(device_id)
                tprint(
                    f"GPU {device_id}: {device_props['name'].decode('utf-8')}, "
                    f"Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB"
                )

            return True
        except Exception as e:
            tprint(f"Error getting GPU device information: {str(e)}")
            SELECTED_GPU_DEVICES = None
            return False

    except ImportError:
        tprint("CuPy not available. Using CPU-only mode.")
        return False
    except Exception as e:
        tprint(f"Error importing GPU libraries: {str(e)}")
        return False


# Initialize the module
import_gpu_libs()
