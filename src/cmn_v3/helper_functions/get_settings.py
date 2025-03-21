#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import importlib.util
from pathlib import Path


def get_settings():
    """
    Load settings from param.py or param_copy.py

    This function tries to load param_copy.py from the output directory if specified in environment variable
    Otherwise, it falls back to the default param.py.

    Returns:
        dict: The settings dictionary from the param module
    """
    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir and os.path.exists(os.path.join(output_dir, "param_copy.py")):
        # Load from output directory
        param_path = os.path.join(output_dir, "param_copy.py")
        spec = importlib.util.spec_from_file_location("param_copy", param_path)
        param_copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(param_copy)
        return param_copy.settings
    else:
        # Fall back to original param.py
        from param import settings

        return settings
