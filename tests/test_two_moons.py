#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from sklearn import datasets
from graph_utils import construct_weightmatrix, predict, graph_grad
from helpers import euclidean_basis, load_models
from wass_dist import wass_dist
from ppoisson import ppoisson

n = 100
k = 2
X,labels = datasets.make_moons(n_samples=n,noise=0.1, random_state = 0)
W = construct_weightmatrix(X)
train_ind = gl.trainsets.generate(labels, rate=5)
train_labels = labels[train_ind]
m = train_ind.size

onehot_train_labels = np.zeros((m, k))
for i in range(train_ind.size):
    onehot_train_labels[i] = euclidean_basis(train_labels[i], k)

mu = np.zeros((n,k))
mu[train_ind] = euclidean_basis(train_labels , k)- train_labels.sum(axis = 0)/train_ind.size

models = load_models("twomoons_data")
potentials = {}

tol = 1e-9
u0 = models[255].u
V0 = graph_grad(u0, W)
u = wass_dist(W, mu, u0, V0, tol = tol, max_iter=1e6)
predictions = predict(u)

plt.scatter(X[:, 0], X[:, 1], c = predict(u))
plt.title(r"Limit initialization. tol = " + "{:.0e}".format(tol))
