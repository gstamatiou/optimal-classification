#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import ot
from ot.datasets import make_1D_gauss as gauss
from ot_class.graph_utils import construct_weightmatrix, graph_grad
from ot_class.wass_dist import wass_dist

# def wass_cost():
        
#     n = 1000  # nb bins
    
#     # bin positions
#     x = np.arange(n, dtype=np.float64)
    
#     # Gaussian distributions
#     a = gauss(n, m=20, s=5)  # m= mean, s= std
#     b = gauss(n, m=60, s=10)
    
#     # loss matrix
#     M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric = 'euclidean')
    
#     res = ot.emd(a, b, M, log = True)
#     return res[1]['cost']

# print(wass_cost())

m = 30
n = 2 * m

mu = np.zeros((n,1))
mu[:m] = np.ones(1)/m
mu[m:] = -np.ones(1)/m

X = np.zeros(n)
X[:m] = np.random.normal(10, 5, m)
X[m:] = np.random.normal(15, 10, m)

plt.scatter(X[:m], np.zeros(m))
plt.scatter(X[m:], np.zeros(m))

W = construct_weightmatrix(X)
u0 = np.random.rand(n,1) * 10

u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-3, 
              max_iter=1e5, verbose=True)

print((u*mu).sum())

a = np.zeros(n)
a[:m] = np.ones(m)/m
b = np.zeros(n)
b[m:] = np.ones(m)/m
M = ot.dist(X.reshape((n,1)), X.reshape((n,1)), metric = 'euclidean')
print(ot.emd(a, b, M, log = True)[1]['cost'])

