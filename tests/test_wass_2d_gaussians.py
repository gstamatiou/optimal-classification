#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from ot_class.wass_dist import wass_dist
from ot_class.graph_utils import construct_weightmatrix, graph_grad
import ot

for m in [20, 40, 60, 80, 100]:
    n = 2*m
    
    X = np.zeros((n,2))
    cov = np.array([[2, -1], [-1, 2]])
    X[:m] = ot.datasets.make_2D_samples_gauss(m, [0,0], np.diag([1,2]), random_state = 0)
    X[m:] = ot.datasets.make_2D_samples_gauss(m, [12,9], np.diag([1,2]), random_state = 1)
    
    plt.scatter(X[:m, 0], X[:m, 1])
    plt.scatter(X[m:, 0], X[m:, 1])
    
    mu = np.zeros((n,1))
    mu[:m] = np.ones(1)/m
    mu[m:] = -np.ones(1)/m
    
    W = construct_weightmatrix(X)
    u0 = np.random.rand(n,1) * 10
    u = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-5,
                    max_iter=1e6, verbose=False, rate = 1)
    
    print((u*mu).sum())
    
    a = np.zeros(n)
    b = np.zeros(n)
    a[:m] = np.ones(1)/m
    b[m:] = np.ones(1)/m
    M = ot.dist(X, X, metric = 'euclidean')
    
    res = ot.emd(a, b, M, log = True)
    print(res[1]['cost'])
    
# The result is 15 (lower bound by jensen, upper bound by W2 distance)