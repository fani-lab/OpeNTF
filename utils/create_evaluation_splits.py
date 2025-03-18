import json
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from utils.json_utils import NumpyArrayEncoder

# Add the project root to the Python path if it's not already there
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_evaluation_splits(
    n_sample, n_folds, train_ratio=0.85, year_idx=None, output="./", step_ahead=1
):
    """
    Create train/validation/test splits for model evaluation.

    Args:
        n_sample: Number of samples in the dataset
        n_folds: Number of cross-validation folds
        train_ratio: Ratio of data to use for training (vs. test)
        year_idx: Temporal indices for time-based splitting
        output: Directory to save the splits
        step_ahead: Number of time steps ahead to predict (for temporal data)

    Returns:
        Dictionary containing the splits
    """
    if year_idx:
        train = np.arange(
            year_idx[0][0], year_idx[-step_ahead][0]
        )  # for teamporal folding, we do on each time interval ==> look at tntf.py
        test = np.arange(year_idx[-step_ahead][0], n_sample)
    else:
        train, test = train_test_split(
            np.arange(n_sample), train_size=train_ratio, random_state=0, shuffle=True
        )

    splits = dict()
    splits["test"] = test
    splits["folds"] = dict()
    skf = KFold(n_splits=n_folds, random_state=0, shuffle=True)
    for k, (trainIdx, validIdx) in enumerate(skf.split(train)):
        splits["folds"][k] = dict()
        splits["folds"][k]["train"] = train[trainIdx]
        splits["folds"][k]["valid"] = train[validIdx]

    with open(f"{output}/splits.json", "w") as f:
        json.dump(splits, f, cls=NumpyArrayEncoder, indent=1)
    return splits
