#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to import GPU libraries and check their availability.
"""

import sys
from pathlib import Path

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.tprint import tprint
from utils.parse_gpus_string import parse_gpus_string

# Globals that will be used by other modules
CUPY_AVAILABLE = False
SELECTED_GPU_DEVICES = None


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
                SELECTED_GPU_DEVICES = parse_gpus_string(SELECTED_GPU_DEVICES)

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


def set_gpu_devices(devices):
    """
    Set the GPU devices to use before initializing GPU libraries.

    Args:
        devices: Can be "all", "first", a list of device indices, or a string to parse

    Returns:
        bool: True if successful, False otherwise
    """
    global SELECTED_GPU_DEVICES
    SELECTED_GPU_DEVICES = devices
    return import_gpu_libs()


# Initialize the module
import_gpu_libs()
