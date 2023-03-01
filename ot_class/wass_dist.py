#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from numpy.random import rand
from graph_utils import graph_grad, graph_div
from pd_alg import power_iteration, pd_alg

def proxF(V, sigma):
    norms = norm(V, axis = 2)

    Z = np.copy(V)
    Z[norms <= sigma] = 0
    Z[norms > sigma] = ((norms[norms > sigma] - sigma)/norms[norms > sigma])[:, np.newaxis] * Z[norms > sigma]
    
    return Z

def proxG(u, tau, mu):
    return tau * mu + u

def wass_dist(W, mu, u0, V0, tol = 1e-2, max_iter = 1e5, verbose = False, rate = 1):
    n = W.shape[0]
    
    K = lambda u: graph_grad(u, W)
    Kstar = lambda V: -graph_div(V, W)
    Kstar_K = lambda u: Kstar(K(u))
    
    _, eig = power_iteration(Kstar_K, rand(n), tol = 1e-4, verbose = False)
    K_norm = np.sqrt(eig)
    sigma = rate/ (2 * K_norm)
    tau = rate/(2 * K_norm )
    
    _proxF = lambda V: proxF(V, sigma)
    _proxG = lambda u: proxG(u, tau, mu)
    
    return pd_alg(_proxF, _proxG, K, Kstar, tau, sigma, u0, V0, 
                  tol = tol, max_iter = max_iter, verbose = verbose)
    