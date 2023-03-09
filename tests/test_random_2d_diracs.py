#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ot_class.wass_dist import wass_dist
from ot_class.graph_utils import construct_weightmatrix, graph_grad
import ot

# =============================================================================
# 
# a_ind = [0, 12, 21]
# b_ind = [-1, -13, -22]
# n = 10*10
# xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
# X = np.array([xv.flatten(), yv.flatten()]).T
# 
# plt.scatter(X[a_ind, 0], X[a_ind, 1], s = 20, label = "a", marker = 'x')
# plt.scatter(X[b_ind, 0], X[b_ind, 1], s = 20, label = "b", marker = 'x')
# plt.scatter(X[:, 0], X[:, 1], s = 2)
# plt.legend()
# 
# mu = np.zeros((n,1))
# mu[a_ind] = 1/3 * np.ones((3,1))
# mu[b_ind] = -1/3 * np.ones((3,1))
# 
# W = construct_weightmatrix(X)
# u0 = np.random.rand(n,1) * 10
# 
# u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-5, 
#                 max_iter=1e6, verbose=True, rate = 1)
# 
# print( (u * mu).sum())
# 
# =============================================================================

m = 20
n = 2*m

np.random.seed(0)
X = np.random.uniform(low = 0, high = 4, size = (n,2))
plt.scatter(X[:m, 0], X[:m, 1])
plt.scatter(X[m:, 0], X[m:, 1])

mu = np.zeros((n,1))
mu[:m] = np.ones(1)/m
mu[m:] = -np.ones(1)/m

W = construct_weightmatrix(X)
u0 = np.random.rand(n,1) * 10
u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-8,
                max_iter=1e6, verbose=False, rate = 1)

print((u*mu).sum())

a = np.zeros(n)
b = np.zeros(n)
a[:m] = np.ones(1)/m
b[m:] = np.ones(1)/m
M = ot.dist(X, X, metric = 'euclidean')

res = ot.emd(a, b, M, log = True)
print(res[1]['cost'])