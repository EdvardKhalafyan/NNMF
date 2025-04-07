import numpy as np
from copy import deepcopy

def apply_symmetric_noise(arr, prob=0.07):
    noisy_arr = deepcopy(arr)
    random_mask = np.random.uniform(size=arr.shape) > (1 - prob)
    noisy_arr[random_mask] = 1 - arr[random_mask]
    return noisy_arr