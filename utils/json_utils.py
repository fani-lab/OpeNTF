import json
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays and data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)
