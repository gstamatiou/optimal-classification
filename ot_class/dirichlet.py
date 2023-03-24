#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph_utils import grad, construct_weightmatrix
from newton import newton
from sklearn import datasets
import graphlearning as gl
from numpy.random import rand, seed
import matplotlib.pyplot as plt
from scipy import sparse

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
    m = g.shape[0]
    
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
    
    
def dirichlet_solve(u0, W, g, p, f, train_ind, tol = 1e-2):
    n = u0.shape[0]
    m = g.shape[0]
    
    W_rearanged = W.copy()
    W_rearanged[np.concatenate((train_ind, np.arange(n, n+m)))] = \
        W_rearanged[np.concatenate((np.arange(n, n+m), train_ind))]
    
    W_rearanged[:, np.concatenate((train_ind, np.arange(n, n+m)))] = \
        W_rearanged[:, np.concatenate((np.arange(n, n+m), train_ind))]
        
    jac = lambda x: dirichlet_jacobian(x, W_rearanged, g.reshape((m,1)), p, f)
    hess = lambda x: dirichlet_hessian(x, W_rearanged, g.reshape((m,1)), p)
    u = newton(u0, lambda x: x, jac, hess, verbose=False, tol=1e-5).reshape(n)
    g = g.reshape(m)
    bigger_u = np.concatenate((u, g))
    bigger_u[np.concatenate((train_ind, np.arange(n, n+m)))] = \
        bigger_u[np.concatenate((np.arange(n, n+m), train_ind))]
        
    return bigger_u

    
# =============================================================================
# n = 4
# m = 2
# p = 8
# train_ind = np.random.randint(low = 0, high = n, size = m)
# labels = (np.random.rand(m) * 10 - 5 * np.ones(m)).round(decimals = 0)
# W = np.random.rand(n,n).round(decimals = 0) * 10
# W = W.T + W
# u = (np.random.rand(n) * 5 - 2.5 * np.ones(n)).round(decimals = 0)
# # =============================================================================
# =============================================================================
# 
# 
# gradu = grad(u)
# normed_gradu = np.abs(gradu)
# A = W * normed_gradu**(p-2)
# for i in range(n):
#     for j in range(n):
#         assert(A[i,j] == W[i,j] * np.abs(u[i] - u[j])**(p-2))
#         
# C = (np.abs(u[:, np.newaxis] - labels))**(p-2)
# for i in range(n):
#     for j in range(m):
#         assert(C[i,j] == (np.abs(u[i] - labels[j]))**(p-2))
# 
# B = W[:, train_ind] * C
# for i in range(n):
#     for j in range(train_ind.shape[0]):
#         assert(B[i,j] == W[i,train_ind[j]] * np.abs(u[i] - labels[j])**(p-2))
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
#         second_term += (W[i,train_ind[j]] * np.abs(u[i] - labels[j])**p)/p
# print(first_term + second_term - dirichlet_energy(u, W, train_ind, labels, p))
# 
# =============================================================================
# =============================================================================
# from scipy.optimize import check_grad
# x0 = (np.random.rand(n)*10 - 5).round(decimals = 0)
# func = lambda x: dirichlet_energy(x, W, train_ind, labels, p)
# func_grad = lambda x: dirichlet_jacobian(x, W, train_ind, labels, p)
# print(check_grad(func, func_grad, x0))
# print(func_grad(x0).shape)
# =============================================================================

# =============================================================================
# #============== Two moons test
# n = 90
# m = 12
# p = 2
# 
# X,labels = datasets.make_moons(n_samples=n+m,noise=0.1, random_state = 0)
# seed(0)
# train_ind = np.sort(np.random.choice(np.arange(n), size=m, replace = False))
# g = labels[train_ind]
# f = np.zeros((n,1))
# W = construct_weightmatrix(X)
# 
# # plt.scatter(X[:,0], X[:,1], alpha=0.5)
# plt.scatter(X[train_ind,0], X[train_ind,1], marker='x', c='r')
# # plt.scatter(X[train_ind,0], X[train_ind,1], c=g, alpha = 0.8)
# 
# bigger_u = dirichlet_solve(rand(n,1), W, g, p, f, train_ind)
# plt.scatter(X[:,0], X[:,1], alpha=0.5, c=np.round(bigger_u))
# print((W*grad(bigger_u)).sum(axis = 1))
# 
# =============================================================================

# =============================================================================
# # ========== MNIST test ====
# from helpers import calculate_accuracy
# n = 1000
# k = 10
# X,labels = gl.datasets.load('mnist')
# X = X[:n]
# labels = labels[:n]
# W = gl.weightmatrix.knn(X,5).toarray()
# 
# rate = 1
# train_ind = gl.trainsets.generate(labels, rate = rate, seed = 1)
# train_labels = labels[train_ind]
# m = train_labels.size
# 
# u = np.zeros((k,n))
# f = np.zeros((n-m,1))
# for i in range(k):
#     g = np.copy(train_labels)
#     g[train_labels == i] = 1
#     g[train_labels != i] = 0
#     
#     u[i] = dirichlet_solve(np.zeros((n-m,1)),W,g,2,f,train_ind)
#     
# predictions = np.argmax(u, axis = 0)
# acc = calculate_accuracy(predictions, labels)
# 
# print(f"Accuracy = {acc*100:.2f}%")
# 
# 
# =============================================================================
