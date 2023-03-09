#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from wass_dist import wass_dist
from graph_utils import construct_weightmatrix, graph_grad
import ot
import graphlearning as gl

def epsilon_matrix(X, epsilon):    
    n = X.shape[0]
    normed_diff = np.apply_along_axis(np.linalg.norm, 2, X[:, np.newaxis] - X)
    W = np.zeros((X.shape[0], X.shape[0]))
    
    for i in range(n):
        for j in range(n):
            if normed_diff[i,j] <= epsilon:
                W[i,j] = 1 - normed_diff[i,j]/epsilon
    
    #W[normed_diff <= epsilon] = 1/epsilon
    
    return W
    
for n in np.arange(10, 35, 3)**2:
    rn = np.sqrt(2)/(2*n)
    sn = np.sqrt(rn)
    print(f"n = {n}, rn = {rn}, s_n = {sn}")
    xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
    X = np.array([xv.flatten(), yv.flatten()]).T
    
    mu = np.zeros((n,1))
    mu[0] = [1]
    mu[-1] = [-1]
    
    W = epsilon_matrix(X, sn)
    #W = construct_weightmatrix(X)
    #W = gl.weightmatrix.epsilon_ball(X, sn).toarray()
    
    
    u0 = np.random.rand(n,1)
    u2d = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-4, 
                    max_iter=1e5, verbose=False)
    print(f"max_grad = {np.abs(graph_grad(u2d, W)).max()}")
    
    fig_pot = plt.figure()
    ax_pot = fig_pot.add_subplot(projection='3d')
    ax_pot.scatter(X[:, 0], X[:, 1], u2d, s = 2)
    plt.show()
    
    print(f"cost: {(u2d * mu).sum()}")
