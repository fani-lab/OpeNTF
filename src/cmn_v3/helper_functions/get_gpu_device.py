#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to get GPU device for use in processing
"""

from .import_gpu_libs import CUPY_AVAILABLE, SELECTED_GPU_DEVICES
from utils.tprint import tprint


def get_gpu_device():
    """
    Get a GPU device for computation

    Returns:
        CuPy device object if GPU is available, None otherwise
    """
    if not CUPY_AVAILABLE:
        return None

    try:
        import cupy as cp

        # Check if we have any GPUs available
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                tprint("No CUDA-capable devices detected")
                return None
        except Exception as e:
            tprint(f"Error detecting CUDA devices: {str(e)}")
            return None

        # Return a device based on SELECTED_GPU_DEVICES
        if SELECTED_GPU_DEVICES is not None:
            if isinstance(SELECTED_GPU_DEVICES, list) and len(SELECTED_GPU_DEVICES) > 0:
                # Use the first device in the selected list
                device_id = SELECTED_GPU_DEVICES[0]
                try:
                    # Validate device_id before creating a Device object
                    if device_id >= device_count:
                        tprint(
                            f"Invalid GPU device {device_id}: device count is {device_count}"
                        )
                        if len(SELECTED_GPU_DEVICES) > 1:
                            # Try next device
                            for alt_id in SELECTED_GPU_DEVICES[1:]:
                                if alt_id < device_count:
                                    device_id = alt_id
                                    tprint(f"Using alternative GPU {device_id}")
                                    break
                            else:
                                tprint("No valid GPU devices found in selection")
                                return None
                        else:
                            # No alternatives, try default device 0
                            if device_count > 0:
                                device_id = 0
                                tprint(f"Using default GPU 0")
                            else:
                                return None

                    device = cp.cuda.Device(device_id)
                    # Check memory to make sure it's usable
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    free_gb = free_mem / (1024**3)
                    total_gb = total_mem / (1024**3)
                    tprint(
                        f"Using GPU {device_id} with {free_gb:.2f}GB free of {total_gb:.2f}GB total"
                    )
                    return device
                except Exception as e:
                    tprint(f"Error selecting GPU device {device_id}: {str(e)}")
                    # Try another device if available
                    if len(SELECTED_GPU_DEVICES) > 1:
                        for alt_id in SELECTED_GPU_DEVICES[1:]:
                            try:
                                if alt_id < device_count:
                                    device = cp.cuda.Device(alt_id)
                                    tprint(f"Using fallback GPU {alt_id}")
                                    return device
                            except:
                                continue

            # If we reach here, either SELECTED_GPU_DEVICES is not a valid list,
            # or we couldn't use any of the specified devices
            try:
                if device_count > 0:
                    tprint("Using default device (0)")
                    return cp.cuda.Device(0)
                else:
                    tprint("No available GPU devices")
                    return None
            except Exception as e:
                tprint(f"Error using default GPU device: {str(e)}")
                return None
        else:
            # No specific device selected, use default (0)
            try:
                if device_count > 0:
                    return cp.cuda.Device(0)
                else:
                    return None
            except Exception as e:
                tprint(f"Error using default GPU device: {str(e)}")
                return None
    except Exception as e:
        tprint(f"Error getting GPU device: {str(e)}")
        return None
