#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from numpy.random import rand
from graph_utils import graph_grad, graph_div, tf_graph_grad, tf_graph_div
from pd_alg import power_iteration, pd_alg
    
def numpy_proxF(V, sigma):
    norms = np.linalg.norm(V, axis = 2)
    Z = np.copy(V)
    Z[norms <= sigma] = 0
    Z[norms > sigma] = ((norms[norms > sigma] - sigma)/norms[norms > sigma])[:, np.newaxis] * Z[norms > sigma]
    
    return Z
   
def proxF(V, sigma):
    norms = tf.linalg.norm(V, axis = 2)
    
    #Z = tf.zeros(V.shape)
    #Z[norms > sigma] = ((norms[norms > sigma] - sigma)/norms[norms > sigma])[:, np.newaxis] * Z[norms > sigma]
    Z = tf.scatter_nd(
        tf.where(norms > sigma), 
        tf.expand_dims((tf.gather_nd(norms, tf.where(norms > sigma)) - sigma)/tf.gather_nd(norms, tf.where(norms > sigma)), axis=1) * tf.gather_nd(V, tf.where(norms > sigma)),
        shape=V.shape)
    
    return Z

def proxG(u, tau, mu):
    return tau * mu + u    
    

def wass_dist(W, mu, u0, V0, tol = 1e-2, max_iter = 1e5, verbose = False, rate = 1):
    n = W.shape[0]
    
    K = lambda u: tf_graph_grad(u, W)
    Kstar = lambda V: -tf_graph_div(V, W)
    Kstar_K = lambda u: Kstar(K(u))
    
    _, eig = power_iteration(Kstar_K, tf.random.uniform(shape = [n,1], minval = 0, maxval=10), tol = 1e-2, verbose = False)
    K_norm = tf.sqrt(eig)
    sigma = rate/ (2 * K_norm)
    tau = rate/(2 * K_norm )
    
    _proxF = lambda V: proxF(V, sigma)
    _proxG = lambda u: proxG(u, tau, mu)
    
    return pd_alg(_proxF, _proxG, K, Kstar, tau, sigma, u0, V0, 
                  tol = tol, max_iter = max_iter, verbose = verbose)
    

# =============================================================================
# # Test tensorflow proxF implementation
# n = 10
# k = 2
# V = tf.Variable(np.arange(n*n*k).reshape((n,n,k)), dtype="float32")
# sigma = 100
# 
# Z = np.zeros((n,n,k))
# for i in range(n):
#     for j in range(n):
#         _norm = tf.linalg.norm(V[i,j])
#         if _norm > sigma:
#             Z[i,j] = (_norm - sigma)/_norm * V[i,j].numpy() 
# 
# print(np.array_equal(Z, proxF(V, sigma).numpy()))
# print(np.array_equal(Z, numpy_proxF(V, sigma)))
# print(np.array_equal(proxF(V,sigma), numpy_proxF(V, sigma)))
# 
# =============================================================================
