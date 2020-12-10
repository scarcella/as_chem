import numpy as np

def split(dataset, splits_p):
    splits = len(dataset)*np.array(splits_p)
    splits = [int(p) for p in list(splits)]
    return splits