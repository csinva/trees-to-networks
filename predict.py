from tqdm import tqdm
import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
from numpy import array as arr
import time

# sklearn models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor

import torch
from torch import nn


'''
do all matrix multiplies in pytorch
'''
def f_torch_basic(net, x):
    t0 = time.perf_counter()
    x = x.reshape(x.shape[0], -1)
    x = net.layers[0](x) 
    t1 = time.perf_counter()
    x[x < 0] = -1
    x[x >= 0] = 1


    t2 = time.perf_counter()
    x = net.layers[1](x)
    x = (x == 0).float()

    x = net.layers[2](x)
    t3 = time.perf_counter()
    print(f't1: {t1-t0:0.2e}, t2: {t2-t1:0.2e} t3: {t3-t2:0.2e}')
    return x

'''
torch with indexing
'''
def f_torch_indexing(x, idxs0, b0, lay1, b1, idxs2):
    t0 = time.perf_counter()

    # lay 1
#     thresh = net.layers[0].bias.data
#     idxs = net.layers[0].weight.data.argmax(dim=1)
    x = x[:, idxs0] + b0 #.shape
    t1 = time.perf_counter()
    x[x < 0] = -1
    x[x >= 0] = 1

    # lay 2
    t2 = time.perf_counter()
    x = lay1(x)

    # lay 3
    x = x.argmax(dim=1)
    preds = idxs2[x]

#     x = x.argmax(dim=1).detach().numpy()
#     preds = np.vectorize(leaf_neuron_num_to_val.get)(x)
    t3 = time.perf_counter()
    print(f't1: {t1-t0:0.2e}, t2: {t2-t1:0.2e} t3: {t3-t2:0.2e}')
    return preds


'''
np using indexing, not mat mults
'''
def f_np_basic(x, idxs0, b0, w1, b1, idxs2):
    # lay 1
    x = x[:, idxs0] + b0
    x[x < 0] = -1
    x[x >= 0] = 1

    # lay 2
    x = x @ w1 + b1
    
    # lay 3
    x = x.argmax(axis=1)
    preds = idxs2[x]
    
    return preds





'''
def m(x):
    x =  x @ neuron_rights + b
    
    return x

m = jit(m, nopython=True, parallel=True)
'''