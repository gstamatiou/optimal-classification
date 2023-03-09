#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import graphlearning as gl
from scipy.optimize import minimize, LinearConstraint
from .graph_utils import degrees
from .energy import penergy, jacobian
from numpy.linalg import norm

def grad_descent(gradient, start, rate, tolerance = 1e-2, max_steps = int(1e9)):
    x = start
    for _ in range(max_steps):
        diff = - rate * gradient(x)

        if norm(diff) < tolerance:
            return x
        x += diff

    return x

class ppoisson():
    def __init__(self, p, W):
        self.p = p
        self.W = W
        self.n = W.shape[0]
        self.u = None
        self.fitted = False
        self.predicted = False
        self.runtime = 0
    
    # train_ind: Indices of (few) labeled points
    # train_labels: labels of labaled points (one hot encoding)
    # start: starting point (n,k) array
    def fit(self, train_ind, train_labels, start = np.zeros(1), method = 'trust-constr'):
        if self.fitted:
            return self.u
        
        start_time = time.time()

        self.k = train_labels.shape[1]
        d = degrees(self.W)
        eye = np.eye(self.k)


        if np.count_nonzero(start) == 0:
            model = gl.ssl.poisson(self.W, solver='gradient_descent')
            integer_coded_train_labels = np.argmax(train_labels, axis = 1)

            start = model.fit(train_ind, integer_coded_train_labels) # model's fit 
                                                                     # doesn't expect one hot encoding
        
        if method == 'trust-constr':
            constrain_matrix = np.concatenate([d[i] * eye for i in range(self.n)], axis = 1)
            linear_constraint = LinearConstraint(constrain_matrix, np.zeros(self.k), np.zeros(self.k))
            
            start = start.flatten()
            res = minimize(penergy, x0 = start, args = (self.W, train_ind, train_labels, self.p), 
                    jac = jacobian, method = 'trust-constr', constraints = linear_constraint)
            
            self.u = res.x.reshape(self.n,self.k)
            self.fitted = True
            
            end_time = time.time()
            self.runtime = (end_time - start_time)/60 # in minutes
    
        elif method == 'grad-desc':
            self.fitted = True
            
            self.u = grad_descent(lambda u: jacobian(u, self.W, train_ind, train_labels, self.p), 
                    start=start, rate = 0.5, tolerance = 1e-2)
            end_time = time.time()
            self.runtime = (end_time - start_time)/60 # in minutes

        return self.u

        
    def predict(self):
        if not self.fitted:
            print("Not fitted yet")
            return -1
        if self.predicted:
            return self.predictions
        
        self.predictions = np.argmax(self.u, axis = 1)
        self.predicted = True

        return self.predictions

    
    def fit_predict(self, train_ind, train_labels, start = np.zeros(1)):
        self.fit(train_ind, train_labels, start)
        
        return self.predict()

    # labels: integer valued
    def accuracy(self, labels):
        if not self.predicted:
            self.predictions = self.predict()
        
        return 1 - np.count_nonzero(labels - self.predictions)/self.n

    def print_info(self):
        info_str = f"########### Gradient Descent (w/ Jacobian) for p = {self.p}\n"\
                        f"\nAccuracy = {self.accuracy() * 100:.2f}%\n"\
                        f"Runtime = {self.runtime:.2f} min"

        print(info_str)
