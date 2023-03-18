#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import graphlearning as gl
from helpers import euclidean_basis, load_models 
from graph_utils import degrees, predict
from numpy.linalg import norm, inv
from energy import hessian, jacobian, penergy
from sklearn import datasets
from ppoisson import ppoisson

def newton_energy(u0_flat, W, train_ind, y, p, rate = 0.2, tol=1e-4, max_iter=1e4, verbose = False):
    n = W.shape[0]
    k = y.shape[1]
    
    D = tf.math.reduce_sum(W, axis = 1)
    # mean = lambda u: (1/D.sum() * (np.diag(D) @ u.reshape((n,k))).sum(axis = 0)).flatten()
    project = lambda u:  u - ((1/tf.reduced_sum(D)) * tf.reduced_sum((tf.linalg.diag(D) @ u), axis = 0))
    u_flat = u0_flat
    
    difference = tol + 1
    it = 0
    while it < max_iter and difference > tol:
        u_flat_bar = u_flat - tf.linalg.inv(hessian(u_flat, W, train_ind, y, p)) @ jacobian(u_flat, W, train_ind, y, p)
        uu_flat = tf.reshape(project(tf.reshape(u_flat_bar, (n,k))), -1)

        it += 1
        difference = tf.linalg.norm(uu_flat - u_flat)
        if verbose:
            print(difference)
            print(u_flat)

        u_flat = uu_flat

    return u_flat

def newton(x0, proj, jac, hess, tol=1e-4, max_iter=1e4, verbose=False):
    x = x0
    
    it = 0
    diff = tol + 1
    while it <= max_iter and diff > tol:
        xx = proj(x - inv(hess(x)) @ jac(x))
        diff = norm(xx - x)
        if verbose:
            print(xx)
            print(diff)
        
        x = xx
    
    if it >= max_iter:
        print("Reached maxinum iterations")
    
    return x
    

# =============================================================================
# 
# p = 8
# n = 100
# k = 2
# X,labels = datasets.make_moons(n_samples=n,noise=0.1, random_state = 0)
# 
# W = gl.weightmatrix.knn(X,10).toarray()
# train_ind = gl.trainsets.generate(labels, rate=5)
# train_labels = labels[train_ind]
# m = train_ind.size
# 
# y = np.zeros((m, k))
# for i in range(train_ind.size):
#     y[i] = euclidean_basis(train_labels[i], k)
# 
# 
# models = load_models("twomoons_data")
# 
# 
# D = degrees(W)
# jac = lambda u : jacobian(u, W, train_ind, y, p)
# hess = lambda u: hessian(u, W, train_ind, y, p)
# proj = lambda u: (u.reshape((n,k)) - ((1/D.sum()) * (np.diag(D) @ u.reshape((n,k))).sum(axis = 0))).flatten()
# 
# u_flat = newton(models[4].u.flatten(), proj, jac, hess, verbose=True)
# 
# # =============================================================================
# # start_time = time.time()
# # u = newton(models[255].u.flatten(), W, train_ind, y, p, verbose=True).reshape((n,k))
# # end_time = time.time()
# # =============================================================================
# 
# plt.scatter(X[:, 0], X[:, 1], c = predict(u))
# 
# print(f"Runtime = {(end_time - start_time)/60:.2f} min")
# =============================================================================


# Test constrained newton minimization by finding the point on a circle closest
# to some other point a

def find_closest():
    tf.random.set_seed(0)
    a = tf.random.uniform([2,1], minval=0, maxval=5)
    jac = lambda x: 2 * (x - a)
    hess = lambda x: tf.eye(x.shape[0])
    proj = lambda x: x / tf.linalg.norm(x)
    sol = newton(tf.random.uniform([2,1]), proj, jac, hess, verbose = True)
    
    fig, ax = plt.subplots()
    ax.axis('equal')
    circle = plt.Circle((0, 0), 1, color='r', fill=False)
    ax.add_patch(circle)
    ax.scatter(a[0], a[1], c = 'blue', label = 'a')
    ax.scatter(sol[0], sol[1], c = 'black', label = 'solution')
    ax.plot([0, a[0]], [0, a[1]], linestyle='dashed')
    ax.legend()
    plt.show()
