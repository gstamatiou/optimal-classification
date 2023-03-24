#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import graphlearning as gl
from helpers import euclidean_basis, calculate_accuracy
from graph_utils import predict
from energy import penergy
from ppoisson import ppoisson
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sample_size = 300
k = 10
X,labels = gl.datasets.load('mnist')
X = X[:sample_size]
labels = labels[:sample_size]
W = gl.weightmatrix.knn(X,5).toarray()

p_vals = [2,5,9,100]
rates = [1,2,3,4,5]

models = {p: {rate: ppoisson(p, W) for rate in rates} for p in p_vals} # models[p][rate] = ppoisson(p, W)
predictions = {p: {rate: np.zeros(sample_size) for rate in rates} for p in p_vals}
#predictions[p][rate][i] = predicted class (0, ..., k) for ith sample using p and rate
accuracy = {p: { rate: 0 for rate in rates } for p in p_vals} #accuracy[p][rate]

for rate in rates:
    train_ind = gl.trainsets.generate(labels, rate = rate, seed = 1)
    train_labels = labels[train_ind]
    euclidean_labels = euclidean_basis(train_labels, 10)
    predictions[2][rate] = models[2][rate].fit_predict(train_ind, euclidean_labels, 
                                                 start = np.zeros(sample_size))
    accuracy[2][rate] = calculate_accuracy(predictions[2][rate], labels)
    info_str = f"########### Gradient Descent for p = 2\n"\
                    f"\nAccuracy = {accuracy[2][rate]*100:.2f}%\n"\
                    f"Runtime = {models[2][rate].runtime:.2f} min"

    print(info_str)

    
for i in range(1, len(p_vals)):
    p = p_vals[i]
    for rate in rates:
        train_ind = gl.trainsets.generate(labels, rate = rate, seed = 1)
        train_labels = labels[train_ind]
        euclidean_labels = euclidean_basis(train_labels, 10)

        predictions[p][rate] = models[p][rate].fit_predict(train_ind, euclidean_labels, 
                                                     start = models[p_vals[i-1]][rate].u.flatten())
    
        accuracy[p][rate] = calculate_accuracy(predictions[p][rate], labels)
        
        info_str = f"########### Gradient Descent for p = {p}\n"\
                        f"\nAccuracy = {accuracy[p][rate]*100:.2f}%\n"\
                        f"Runtime = {models[p][rate].runtime:.2f} min"
    
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