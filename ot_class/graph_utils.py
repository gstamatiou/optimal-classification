#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

# Calculates vector of degrees for the weight matrix W
# W: (n,k) numpy array
def degrees(W):
    return W.sum(axis = 1)

# Example
# Calculates degrees of equilateral triangle
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# print(degrees(W))

def graph_div(V, W):
    n = V.shape[0]
    k = V.shape[2]
    div = ( V * (W[:, :, np.newaxis] + np.zeros((n,n,k))) ).sum(axis = 1)
    return div

def tf_graph_div(V,W):
    return tf.reduce_sum(tf.expand_dims(W, axis = 2) * V, axis = 1)
   
"""
Computes the weighted gradient of u
u: (n,k) matrix
W: (n,n) matrix

Returns:
gradu: (n, n, k)

where gradu[i, j, l] = W[i,j] * u[j,l]-u[i,l]
"""
def graph_grad(u, W):
    gradu = W[:, :, np.newaxis] * (-u[:, np.newaxis] + u)

    return gradu


def tf_graph_grad(u, W):
    gradu = np.expand_dims(W, axis = -1) * (-tf.expand_dims(u, axis=1) + u)

    return gradu

# =============================================================================
# # Test tf_graph_div, tf_graph_grad
# n = 10
# k = 3
# V = tf.random.uniform([n,n,k], minval = -10, maxval = 10)
# W = tf.random.uniform([n,n], minval = 0, maxval = 10)
# _div = np.zeros((n,k))
# 
# for i in range(n):
#     for j in range(n):
#         _div[i] += W[i,j].numpy() * V[i,j].numpy()
# 
# print(np.max(np.abs(_div- tf_graph_div(V, W).numpy())))
# 
# u = tf.random.uniform([n,k], minval = -10, maxval = 10)
# print(np.max(np.abs(graph_grad(u.numpy(), W) - tf_graph_grad(u, W))))
# =============================================================================

"""
Computes the non-weighted gradient of u
u: (n,k) matrix

Returns:
gradu: (n, n, k)

where gradu[i, j, l] = u[j,l]-u[i,l]
"""

def grad(u):
    gradu = -u[:, np.newaxis] + u

    return gradu

# # Example:
# u = np.arange(6).reshape(3,2)
# print("u is\n", u)
# print("\nAnd gradu is")
# for i in range(u.shape[0]):
#     for j in range(u.shape[1]):
#         print(graph_grad(u)[i,j], end = "\t\t")
#     print()

# Returns the labels that u predicts
def predict(u):
    return np.argmax(u, axis = 1)

# X: (n,) or (n,k) matrix
# Returns an (n,n)-matrix A whose non-diagonal values are:
#   A[i,j] = 1/norm(X[i] - X[j])
# and whose diagonal values are A[i,i] = 1
def construct_weightmatrix(X): 
    n = X.shape[0]
    
    #diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis] - X
    
    
    if X.ndim == 1:
        normed_diff = np.abs(diff)
    else:
        normed_diff = np.apply_along_axis(norm, 2, diff)
    
    normed_diff[np.diag_indices(n)] = np.ones(n)
    
    return 1/normed_diff

# =============================================================================
# Calculates the supremum norm of the graph_grad of u
# u: (n,k)-array
# W: (n,n)-array
# =============================================================================
def sup_graph_grad(u, W):
    print(np.apply_along_axis(norm, 2, graph_grad(u, W)).max())