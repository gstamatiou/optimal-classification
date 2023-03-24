#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import graphlearning as gl
from helpers import euclidean_basis, calculate_accuracy
from graph_utils import predict
from dirichlet import dirichlet_solve
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from helpers import calculate_accuracy
from time import time

n = 800
k = 10
X,labels = gl.datasets.load('mnist')
X = X[:n]
labels = labels[:n]
W = gl.weightmatrix.knn(X,5).toarray()

p_vals = [2,5,9, 15]
rates = [1,2,3,4,5]

u = {p: {rate: np.zeros((k,n)) for rate in rates} for p in p_vals }
accuracy = {p: { rate: 0 for rate in rates } for p in p_vals} #accuracy[p][rate]


for rate in rates:
    start_time = time()
    train_ind = gl.trainsets.generate(labels, rate = rate, seed = 2)
    train_labels = labels[train_ind]
    m = train_labels.size #amount of labeled points
    
    f = np.zeros((n-m,1))
    for i in range(k):
        g = np.copy(train_labels)
        g[train_labels == i] = 1
        g[train_labels != i] = 0
        
        u[2][rate][i] = dirichlet_solve(np.zeros((n-m,1)),W,g,2,f,train_ind, tol=1e-6)
        
    predictions = np.argmax(u[2][rate], axis = 0)
    accuracy[2][rate] = calculate_accuracy(predictions, labels)
    end_time = time()
    runtime = (end_time - start_time)/60
    
    info_str = f"########### p = 2, rate = {rate}\n"\
                    f"Accuracy = {accuracy[2][rate]*100:.2f}%\n"\
                    f"Runtime = {runtime:.2f} min"

    print(info_str)

for j in range(1, len(p_vals)):
    p = p_vals[j]
    for rate in rates:
        start_time = time()
        train_ind = gl.trainsets.generate(labels, rate = rate, seed = 2)
        train_labels = labels[train_ind]
        not_train_ind = np.delete(np.arange(n), train_ind)
        m = train_labels.size #amount of labeled points
        
        f = np.zeros((n-m,1))
        for i in range(k):
            g = np.copy(train_labels)
            g[train_labels == i] = 1
            g[train_labels != i] = 0
            
            u[p][rate][i] = dirichlet_solve(u[p_vals[j-1]][rate][i, not_train_ind].reshape((n-m, 1)),W,g,p,f,train_ind, tol = 1e-6)
            
        predictions = np.argmax(u[p][rate], axis = 0)
        accuracy[p][rate] = calculate_accuracy(predictions, labels)
        end_time = time()
        runtime = (end_time - start_time)/60
        
        info_str = f"########### p = {p}, rate = {rate}\n"\
                        f"Accuracy = {accuracy[p][rate]*100:.2f}%\n"\
                        f"Runtime = {runtime:.2f} min"
    
        print(info_str)

fig,ax = plt.subplots()

ax.set_xlabel("Num. of labels per class")
ax.set_ylabel("Accuracy")
plt.xticks(ticks = rates, labels = rates)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(0, 1)
plt.grid(linestyle="--")

for p in p_vals:
    ax.plot(rates, list(accuracy[p].values()), '-o', label=f"$p = {p}$")
    
plt.legend()
plt.show()