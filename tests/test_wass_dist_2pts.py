#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wass_dist import wass_dist
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import construct_weightmatrix, graph_grad
import ot, ot.plot

n = 100+1

X = np.linspace(0, 2, n)
W = construct_weightmatrix(X)

mu = np.zeros((n,1))
mu[0] = [1/3]
mu[int((n-1)/2)] = [1/3-1/2]
mu[-1] = [1/3 -1/2]

plt.scatter([X[0],  X[int((n-1)/2)], X[-1]], np.zeros(3), c = 'b')
plt.scatter([X[int((n-1)/2)], X[-1]], np.zeros(2), c = 'r', marker='x')
u1d = wass_dist(W, mu, np.zeros((n,1)), np.zeros((n,n,1)), 
                      tol = 1e-5, max_iter=1e5, verbose=False)

print((u1d * mu).sum())
plt.scatter(X, u1d, s = 1)

a = np.zeros(n)
b = np.zeros(n)
a[0], a[int((n-1)/2)], a[-1] = 1/3, 1/3, 1/3
b[int((n-1)/2)], b[-1] = 1/2, 1/2
#print(ot.emd(a, b, W, log=True)[1]['cost'])
print(ot.emd(a, b, ot.dist(X.reshape((n, 1)), X.reshape((n, 1)), metric = 'euclidean'), log=True)[1]['cost'])

# u1d = wass_dist(ot.dist(X.reshape((n, 1)), X.reshape((n, 1))), mu, np.random.rand(n,1)*10, np.random.rand(n,n,1)*10, 
#                       tol = 1e-7, max_iter=1e5, verbose=False)

# print((u1d * mu).sum())
