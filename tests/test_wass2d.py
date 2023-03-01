#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from wass_dist import wass_dist
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import construct_weightmatrix, graph_grad

res = dict()

for i in range(2, 16):
    n = i ** 2
    
    xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
    X = np.array([xv.flatten(), yv.flatten()]).T
    
    mu = np.zeros((n,1))
    mu[0] = [1]
    mu[-1] = [-1]
    W = construct_weightmatrix(X)
    u0 = np.random.rand(n,1)
    u2d = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-5, 
                    max_iter=1e5, verbose=False)
    
    
    # fig_pot = plt.figure()
    # ax_pot = fig_pot.add_subplot(projection='3d')
    # ax_pot.scatter(X[:, 0], X[:, 1], u2d, s = 2)
    # plt.show()
    
    res[i] = np.abs((u2d * mu).sum() - np.sqrt(2))

plt.plot(list(res.keys()), list(res.values()), '.r-')
