import numpy as np


def average_cross_arrays(arrays):
    max_length = max(len(arr) for arr in arrays)
    padded_arrays = np.array(
        [
            np.pad(arr, (0, max_length - len(arr)), mode="constant", constant_values=-1)
            for arr in arrays
        ]
    )
    masked_array = np.ma.masked_where(padded_arrays == -1, padded_arrays)
    return masked_array.mean(axis=0)
