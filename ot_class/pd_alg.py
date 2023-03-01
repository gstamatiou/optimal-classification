#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

def power_iteration(K, x_0, tol = 1e-3, num_iterations = 10000000, verbose = False):
    b = x_0/norm(x_0)
    for _ in range(num_iterations):
        bb = K(b)/norm(K(b))
        diff = norm(b - bb)
        
        if verbose:
            print("PowerIteration:", diff)
            
        if diff < tol:
            return bb, norm(K(bb))/norm(bb)
        b = np.copy(bb)

    return bb, norm(K(bb))/norm(bb)

def pd_alg(proxF, proxG, K, Kstar, tau, sigma, x0, y0, tol = 1e-2, 
           max_iter = 1e6, verbose = False):
    y = y0
    x = x0
    x_bar = x0
    
    difference = tol + 1
    it = 0
    while (it <= max_iter and difference > tol):
        yy = proxF(y + sigma * K(x_bar))
        xx = proxG(x - tau * Kstar(yy))
        xx_bar = 2 * xx - x

        difference = np.max(np.abs(xx - x))
        it += 1

        x = xx
        x_bar = xx_bar
        y = yy

        if verbose:
            print(difference)
    
    if it >= max_iter:
        print("reached max iterations with diff:", difference)
    return x
