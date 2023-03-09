#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from graph_utils import grad


def dirichlet_jacobian(u, W, train_ind, labels, p):
    gradu = grad(u)
    normed_gradu = np.abs(gradu)
    
    A = W * normed_gradu**(p-2)
    C = np.abs(u[:, np.newaxis] - labels) # C[i,j] = |u[i] - labels[j]|    
    B = W[:, train_ind] * C
    D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
    
    L = D - A
    
    jac = L @ u - B @ labels
    
    return jac

def dirichlet_hessian(u, W, train_ind, labels, p):
    gradu = grad(u)
    normed_gradu = np.abs(gradu)
    
    A = W * normed_gradu**(p-2)
    C = np.abs(u[:, np.newaxis] - labels) # C[i,j] = |u[i] - labels[j]|    
    B = W[:, train_ind] * C
    D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
    
    L = D - A
    
    hess = (p-1) * L
    
    return hess

n = 40
m = 3
p = 8
train_ind = np.random.randint(low = 0, high = n, size = m)
labels = np.random.rand(m) * 10 - 5 * np.ones(m)
W = np.random.rand(n,n) * 10
W = W.T + W
u = np.random.rand(n) * 10 - 5 * np.ones(n)


gradu = grad(u)
normed_gradu = np.abs(gradu)
A = W * normed_gradu**(p-2)
for i in range(n):
    for j in range(n):
        assert(A[i,j] == W[i,j] * np.abs(u[i] - u[j])**(p-2))
        
C = np.abs(u[:, np.newaxis] - labels)
for i in range(n):
    for j in range(m):
        assert(C[i,j] == np.abs(u[i] - labels[j]))

B = W[:, train_ind] * C
for i in range(n):
    for j in range(train_ind.shape[0]):
        assert(B[i,j] == W[i,train_ind[j]] * np.abs(u[i] - labels[j]))

D = np.diag(B.sum(axis = 1) + A.sum(axis = 1))
for i in range(n):
    for j in range(n):
        if i != j:
            assert(D[i,j] == 0)
        else:
            assert(D[i,j] == A[i].sum() + B[i].sum())
