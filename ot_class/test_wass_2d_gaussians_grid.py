#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from wass_dist import wass_dist
from graph_utils import construct_weightmatrix, graph_grad, grad, tf_graph_grad
import ot
import graphlearning as gl
import tensorflow as tf

def epsilon_matrix(X, epsilon):    
    n = X.shape[0]
    normed_diff = np.apply_along_axis(np.linalg.norm, 2, X[:, np.newaxis] - X)
    W = np.zeros((X.shape[0], X.shape[0]))
    
    # for i in range(n):
    #     for j in range(n):
    #         if normed_diff[i,j] <= epsilon:
    #             W[i,j] = 1 - normed_diff[i,j]/epsilon
    
    W[normed_diff <= epsilon] = 1/epsilon
    
    return W

def eta(t):
    if t > 1:
        return 0
    else:
        return 4*(1-t)

def slow_epsilon_matrix(X, epsilon):
    n = X.shape[0]
    W = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            W[i,j] = 1/epsilon*eta(np.linalg.norm(X[i]-X[j])/epsilon)
            
    return W

    
# =============================================================================
# for exponent in [1/9, 1/10, 1/11]:    
#     for n in np.arange(5, 41, 5)**2:
#         xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
#         X = np.array([xv.flatten(), yv.flatten()]).T
#     
#         rn = np.sqrt(2*np.linalg.norm(X[0] - X[1])**2)/2
#         sn = rn**exponent
#         
#         print(f"n = {n}\t rn = {rn:.2f}\t s_n = {sn:.2f} = rn^{np.log(sn)/np.log(rn):.2f}\t", 
#               end=' ')
#         
#         mu = np.zeros((n,1))
#         mu[0] = [1]
#         mu[-1] = [-1]
#         
#         W = slow_epsilon_matrix(X, sn)
#         #W = construct_weightmatrix(X)
#         #W = gl.weightmatrix.epsilon_ball(X, sn).toarray()
#         num_neighbors = np.count_nonzero(W[int(n/2)])
#         if num_neighbors == 1:
#             sn = 3 * sn
#             
#         print(f"neighbors = {np.count_nonzero(W[int(n/2)])}\t", end='')
#         
#         u0 = np.random.rand(n,1)
#         u2d = wass_dist(W, mu, u0, -grad(u0), tol = 1e-6, 
#                         max_iter=1e5, verbose=False)
#         #print(f"max_grad = {np.abs(graph_grad(u2d, W)).max()} ", end = '')
#         
#         fig_pot = plt.figure()
#         ax_pot = fig_pot.add_subplot(projection='3d')
#         ax_pot.scatter(X[:, 0], X[:, 1], u2d, s = 2)
#         plt.show()
#         
#         print(f"cost: {(u2d * mu).sum():.4f}")
# 
# =============================================================================
exponent = 1/12
res = dict()
k = 1
for n in np.arange(5, 35, 3)**2:
    xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
    X = np.array([xv.flatten(), yv.flatten()]).T

    rn = np.sqrt(2*np.linalg.norm(X[0] - X[1])**2)/2
    sn = rn**exponent
    
    print(f"n = {n}\t rn = {rn:.2f}\t s_n = {sn:.2f} = rn^{np.log(sn)/np.log(rn):.2f}\t", 
          end=' ')
    
    mu = tf.scatter_nd(tf.constant([[0],[n-1]]), tf.constant([[1],[-1]], dtype="float32"), shape=[n,1])

    
    W = slow_epsilon_matrix(X, sn)
    #W = construct_weightmatrix(X)
    #W = gl.weightmatrix.epsilon_ball(X, sn).toarray()

        
    print(f"neighbors = {np.count_nonzero(W[int(n/2)])}\t", end='')
    
    u0 = np.random.rand(n,1)
    u2d = wass_dist(tf.constant(W, dtype="float32"), mu, tf.constant(u0, dtype="float32"), tf.zeros([n,n,k], dtype="float32"), tol = 1e-7, 
                    max_iter=1e5, verbose=False)
    #print(f"max_grad = {np.abs(graph_grad(u2d, W)).max()} ", end = '')
    
    fig_pot = plt.figure()
    ax_pot = fig_pot.add_subplot(projection='3d')
    ax_pot.scatter(X[:, 0], X[:, 1], u2d, s = 2)
    plt.show()
    
    res[n] = tf.reduce_sum(u2d * mu)
    print(f"cost: {res[n]:.4f}")
    
plt.plot(list(res.keys()), list(res.values()), 'o-')