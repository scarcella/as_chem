import numpy as np

def split(dataset, splits_p):
    splits = len(dataset)*np.array(splits_p)
    splits = [int(p) for p in list(splits)]
    if sum(splits) > len(dataset):
        splits[0] = splits[0] -1 
    elif sum(splits) < len(dataset):
         splits[1] = splits[1] + 1 
    return splits