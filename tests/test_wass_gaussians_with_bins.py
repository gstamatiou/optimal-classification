#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from graph_utils import construct_weightmatrix, graph_grad
from wass_dist import wass_dist

    
n = 150  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=30, s=5)  # m= mean, s= std
b = gauss(n, m=50, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), metric = 'euclidean')

res = ot.emd(a, b, M, log = True)
print(res[1]['cost'])

mu = a - b


W = construct_weightmatrix(x)
u0 = np.random.rand(n,1) * 10

u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-5, 
                max_iter=1e4, verbose=True, rate = 0.1)

print((u*mu).sum())
# a = np.zeros(2*n)
# a[:n] += np.ones(n)
# b = np.zeros(2*n)
# b[n:] += np.ones(n)
# print(ot.emd(a, b, W, log = True)[1]['cost'])

