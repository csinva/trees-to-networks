import time
import torch
import numpy as np

def t(func, num_reps=1):
    ts = []
    for num_rep in range(num_reps):
        s = time.perf_counter()
        func()
        ts.append(time.perf_counter() - s)
    return np.mean(ts)

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())