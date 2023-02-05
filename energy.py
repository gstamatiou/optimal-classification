#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graph_utils import grad
import numpy as np

"""
u_flattened: n*k matrix
W: weight matrix
y: (m,k) matrix
idx = vector of labelled indices
p = p constant for laplace operator
"""
def penergy(u_flattened, W, idx, y, p):
    k = y.shape[1]
    n = W.shape[0]
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    y_bar = (1/y.shape[0]) * y.sum(axis = 0)
    
    first_summand = (1/(2*p)) * (W * (np.apply_along_axis(np.linalg.norm, 2, gradu) ** p)).sum()
    second_summand = np.sum( (y - y_bar) * u[idx] )

    return first_summand - second_summand

# ## Example (see pdf)
# u = np.array([[1,0], [0,1], [1/2, 1/2]])
# print(graph_grad(u))
# y = np.array([[1,0], [0,1]])
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# idx = [0, 1]
# p = 2

# #print(np.apply_along_axis(np.linalg.norm, 2, graph_grad(u)))
# print("2-energy is", penergy(u, W, idx, y, 2))
# print("3-energy is", penergy(u, W, idx, y, 3))

def jacobian(u_flattened, W, idx, y, p):
    n = W.shape[0]
    k = u_flattened.size//n
    m = y.shape[0]    
    
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    normed_gradu = np.apply_along_axis(np.linalg.norm, 2, gradu)
    y_bar = y.sum(axis = 0)/m
    
    if p > 2:
        # ... = w*normed_gradu**(p-2)
        a1 = W * (normed_gradu**(p-2)) + np.zeros((k, n, n)) # a1[s,i,r] = ...[i, r]
        A = np.transpose(a1, (1, 2, 0)) # A[i,r, s] = ...[i, r]

        u1 = np.zeros( (n, n, k) ) + u #u1[i, r, s] = u[r, s]

        u21 = np.zeros( (n,n,k) ) + u # u21[r, i, s] = u[i, s]
        u2 = np.swapaxes(u21, 0, 1) # u2[i, r, s] = u[i, s]

        C = u1 - u2

        B = np.zeros( (n, k) )
        B[idx] = y - y_bar

        jac = (A * C).sum(axis = 0) - B
    
    elif p == 2:
        a1 = W + np.zeros((k, n, n)) # a1[s,i,r] = W[i, r]
        A = np.transpose(a1, (1, 2, 0)) # A[i, r, s] = W[i, r]
        
        u1 = np.zeros( (n, n, k) ) + u #u1[i, r, s] = u[r, s]

        u21 = np.zeros( (n,n,k) ) + u # u2[r, i, s] = u[i, s]
        u2 = np.swapaxes(u21, 0, 1)

        C = u1 - u2

        B = np.zeros( (n, k) )
        B[idx] = y - y_bar
            
        jac = (A * C).sum(axis = 0) - B
    
    return jac.flatten()
    
# # Testing vectorized jacobian
# p = 6
# n = 15
# k = 2

# X,labels = datasets.make_moons(n_samples=n,noise=0.1)
# W = gl.weightmatrix.knn(X,10).toarray()
# train_ind = gl.trainsets.generate(labels, rate=5)
# train_labels = labels[train_ind]
# m = train_ind.size

# y = np.zeros((m, k))
# for i in range(train_ind.size):
#     y[i] = euclidean_basis(train_labels[i], k)
    

# u = np.random.random(size = (n,k))

# scipy.optimize.check_grad(penergy, jacobian, u.flatten(), W, train_ind, y, p)
