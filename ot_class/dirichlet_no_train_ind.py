#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph_utils import grad
from numpy.random import rand, seed
from sklearn import datasets
import graphlearning as gl
import matplotlib.pyplot as plt
from newton import newton
from graph_utils import construct_weightmatrix, predict, graph_grad
from scipy.optimize import minimize

def dirichlet_jacobian(u, W, g, p, f):
    n = u.shape[0]
    m = g.shape[0]
    gradu = grad(u).reshape((n,n))
    normed_gradu = np.abs(gradu)
    
    A =( W[:n,:n] * normed_gradu**(p-2)).reshape((n,n))
    C = (np.abs(u[:, np.newaxis] - g)**(p-2)).reshape((n,m)) # C[i,j] = |u[i] - g[j]|**(p-2)    
    B = (W[:n, n:] * C).reshape((n,m))
    D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
    
    L = D - A
    
    jac = np.matmul(L, u.reshape(n,1)) - np.matmul(B, g.reshape(m,1)) + f
    
    return jac

def dirichlet_hessian(u, W, g, p):
    n = u.shape[0]
    gradu = grad(u).reshape((n,n))
    normed_gradu = np.abs(gradu)
    
    A = (W[:n, :n] * normed_gradu**(p-2)).reshape((n,n))
    C = (np.abs(u[:, np.newaxis] - g)**(p-2)).reshape((n,m)) # C[i,j] = |u[i] - labels[j]|^(p-2)    
    B = W[:n,n:] * C
    D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
    
    L = D - A
    
    hess = (p-1) * L
    
    return hess

def dirichlet_energy(u, W, g, p, f):
#    n = u.shape[0]
# =============================================================================
#     E = W[:n,:n] * np.abs(u[:, np.newaxis] - u)**p #A[i,j] = W[i,j] * |u[i] - u[j]|^p
#     F = W[:n,n:] * np.abs(u[:, np.newaxis] - g)**p 
#         # B[i,j] = W[i,j+n] * |u[i] - g[j]|^p
#     return E.sum()/(2*p) + F.sum()/p  
# =============================================================================
    bigger_u = np.concatenate((u, g), axis=0)
    H = W * np.abs(bigger_u[:, np.newaxis] - bigger_u)**p
    
    return H.sum()/(2*p) + (f*u).sum()
    
    

# =============================================================================
# n = 4
# m = 2
# p = 8
# seed(0)
# g = (rand(m) * 10 - 5 * np.ones(m)).round(decimals = 0)
# seed(1)
# W = rand(n+m,n+m).round(decimals = 0) * 10
# W = W.T + W
# seed(4)
# u = (rand(n) * 5 - 2.5 * np.ones(n)).round(decimals = 0)
# 
# 
# gradu = grad(u)
# normed_gradu = np.abs(gradu)
# A = W[:n,:n] * normed_gradu**(p-2)
# for i in range(n):
#     for j in range(n):
#         assert(A[i,j] == W[i,j] * np.abs(u[i] - u[j])**(p-2))
# 
# C = np.abs(u[:, np.newaxis] - g)**(p-2) # C[i,j] = |u[i] - g[j]|**(p-2)
# for i in range(n):
#     for j in range(m):
#         assert(C[i,j] == (np.abs(u[i] - g[j]))**(p-2))
# 
# B = W[:n, n:] * C
# for i in range(n):
#     for j in range(m):
#         assert(B[i,j] == W[i,j+n] * np.abs(u[i] - g[j])**(p-2))
# 
# D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
# for i in range(n):
#     for j in range(n):
#         if i != j:
#             assert(D[i,j] == 0)
#         else:
#             assert(D[i,j] == A[i].sum() + B[i].sum())
# 
# =============================================================================

# =============================================================================
# first_term = 0
# for i in range(n):
#     for j in range(n):
#         first_term += (W[i,j] * np.abs(u[i] - u[j])**p)/(2*p)
# second_term = 0
# for i in range(n):
#     for j in range(m):
#         second_term += (W[i,j+n] * np.abs(u[i] - g[j])**p)/p
# print(first_term + second_term - dirichlet_energy(u, W, g, p))
# 
# =============================================================================
# =============================================================================
# 
# from scipy.optimize import check_grad
# seed(3)
# x0 = (rand(n)*10 - 5).round(decimals = 0)
# func = lambda x: dirichlet_energy(x, W, g, p)
# func_grad = lambda x: dirichlet_jacobian(x, W, g, p)
# print(check_grad(func, func_grad, x0))
# print(func_grad(x0).shape)
# =============================================================================


# =============================================================================
# n = 99
# m = 2
# p = 2
# p_vals = [[2, 4], [8, 16], [32, 64]] + [[100 + 10 * i, 105 + 10 * i]  for i in range(0,16)]
# 
# # Generate training data and label sets
# X = np.linspace(0, 1, n+m)
# 
# 
# train_ind = np.array([40, 80])
# g = np.array([1, 0])
# f = np.zeros((n,1))
# 
# X[np.concatenate((train_ind, np.arange(n, n+m)))] = \
#     X[np.concatenate((np.arange(n, n+m), train_ind))]
# 
# W = gl.weightmatrix.knn(X.reshape((n+m,1)),10).toarray()
# 
# plt.scatter(X, np.zeros(n+m), alpha = 0.5, s=1)
# plt.scatter(X[n:], np.zeros(m), marker = 'x', c='r')
# 
# jac = lambda x: dirichlet_jacobian(x, W, g.reshape((m,1)), p, f)
# hess = lambda x: dirichlet_hessian(x, W, g.reshape((m,1)), p)
# u = newton(rand(n,1), lambda x: x, jac, hess, verbose=False, tol=1e-7).reshape(n)
# g = g.reshape(m)
# bigger_u = np.concatenate((u, g))
# plt.scatter(X, bigger_u,s = 1)
# assert(np.max((W*grad(bigger_u)).sum(axis=1)[:n]) < 1e-7) #Check that Δu = 0
# =============================================================================



# Two moons test
n = 90
m = 12
p = 2

X,labels = datasets.make_moons(n_samples=n+m,noise=0.1, random_state = 0)
seed(0)
train_ind = np.sort(np.random.choice(np.arange(n), size=m, replace = False))
g = labels[train_ind]
f = np.zeros((n,1))
X[np.concatenate((train_ind, np.arange(n, n+m)))] = \
    X[np.concatenate((np.arange(n, n+m), train_ind))]
W = construct_weightmatrix(X)

# plt.scatter(X[:,0], X[:,1], alpha=0.5)
plt.scatter(X[n:,0], X[n:,1], marker='x', c='r')
# plt.scatter(X[n:,0], X[n:,1], c=g, alpha = 0.8)

jac = lambda x: dirichlet_jacobian(x, W, g.reshape((m,1)), p, f)
hess = lambda x: dirichlet_hessian(x, W, g.reshape((m,1)), p)
u = newton(rand(n,1), lambda x: x, jac, hess, verbose=True, tol=1e-7).reshape(n)
g = g.reshape(m)
bigger_u = np.concatenate((u, g))

plt.scatter(X[:,0], X[:,1], alpha=0.5, c=np.round(bigger_u))



# =============================================================================
# # MNIST test
# n = 100
# m = 10
# p = 2
# 
# X, labels = gl.datasets.load('mnist')
# X = X[:(n+m)]
# labels = labels[:(n+m)]
# train_ind = np.sort(gl.trainsets.generate(labels[:n], rate=1,seed=0))
# 
# f = np.zeros((n,1))
# X[np.concatenate((train_ind, np.arange(n, n+m)))] = \
#     X[np.concatenate((np.arange(n, n+m), train_ind))]
# labels[np.concatenate((train_ind, np.arange(n, n+m)))] = \
#     labels[np.concatenate((np.arange(n, n+m), train_ind))]
# W = construct_weightmatrix(X)
# 
# res = np.zeros((10, n+m))
# for i in range(10):
#     g = labels[n:]
#     g[g != i] = 0
#     g[g == i] = 1
#     jac = lambda x: dirichlet_jacobian(x, W, g.reshape((m,1)), p, f)
#     hess = lambda x: dirichlet_hessian(x, W, g.reshape((m,1)), p)
#     res[i] = newton(rand(n,1), lambda x: x, jac, hess, verbose=True, tol=1e-7).reshape(n)
# 
# pred_labels = np.argmax(u, axis = 0)
# accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,len(train_ind))
# print(accuracy)
# # plt.scatter(X[:,0], X[:,1], alpha=0.5, c=np.round(bigger_u))
# =============================================================================
