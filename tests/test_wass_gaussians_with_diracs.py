#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from graph_utils import construct_weightmatrix, graph_grad
from wass_dist import wass_dist

def wass_cost():
        
    n = 1000  # nb bins
    
    # bin positions
    x = np.arange(n, dtype=np.float64)
    
    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std
    b = gauss(n, m=60, s=10)
    
    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric = 'euclidean')
    
    res = ot.emd(a, b, M, log = True)
    return res[1]['cost']

print(wass_cost())
for n in [50, 100, 200, 250, 300]:
    mu = np.zeros((2*n,1))
    mu[:n] = 1/n
    mu[n:] = -1/n
    
    X = np.zeros(2*n)
    X[:n] = np.random.normal(20, 5, n)
    X[n:] = np.random.normal(60, 10, n)
    
    W = construct_weightmatrix(X)
    u0 = np.random.rand(2 * n,1) * 10
    
    u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-3, 
                    max_iter=1e5, verbose=True)
    
    print((u*mu).sum())
    # a = np.zeros(2*n)
    # a[:n] += np.ones(n)
    # b = np.zeros(2*n)
    # b[n:] += np.ones(n)
    # print(ot.emd(a, b, W, log = True)[1]['cost'])
    
