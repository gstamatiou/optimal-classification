#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def epsilon_matrix(X, epsilon):
    n = X.shape[0]
    pair_dist = pairwise_distances(X)
    W = np.zeros((n,n))
    
    W[pair_dist < epsilon] = 1/epsilon * 4 * (1 - pair_dist[pair_dist < epsilon]/epsilon)
    
    
    return W

n_vals = np.arange(3, 25, 1)**2
errors = {n: 0 for n in n_vals}
for n in n_vals:
    np.random.seed(1)
    X = np.random.rand(n-2, 2)
    X = np.insert(X, 0, np.array([0,0]), axis = 0)
    X = np.vstack((X, np.array([1,1])))
    
    # |X[0] - X[1]| = 1/(sqrt(n) - 1) for regular grid
    rn = np.sqrt(2*(1/(np.sqrt(n) - 1))**2)/2
    sn = rn**0.5
    
    print(f"n = {n}", end=' ')
    
    W = epsilon_matrix(X, sn)
    
    num_neighbors = np.count_nonzero(W[int(n/2)])
    if num_neighbors == 1:
        sn = 3 * sn
        
    distances = np.inf * np.ones(W.shape)
    distances[W != 0] = 1/W[W != 0]
    dist_matrix = dijkstra(csgraph=csr_matrix(distances), directed=False, indices=0, return_predecessors=False)
    print(dist_matrix[-1])
    errors[n] = np.abs(dist_matrix[-1] - np.sqrt(2))
   
fig,ax = plt.subplots()

ax.set_xlabel("Num. of grid points")
ax.set_ylabel("Error: $|\sqrt{2} - \mathrm{GraphDistance}|$")

ax.semilogy(list(errors.keys()), list(errors.values()), "-o", label = "Random grid")

for n in n_vals:
    xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
    X = np.array([xv.flatten(), yv.flatten()]).T
    
    # |X[0] - X[1]| = 1/(sqrt(n) - 1) for regular grid
    rn = np.sqrt(2*(1/(np.sqrt(n) - 1))**2)/2
    sn = rn**0.5
    
    print(f"n = {n}", end=' ')
    
    W = epsilon_matrix(X, sn)
    
    num_neighbors = np.count_nonzero(W[int(n/2)])
    if num_neighbors == 1:
        sn = 3 * sn
        
    distances = np.inf * np.ones(W.shape)
    distances[W != 0] = 1/W[W != 0]
    dist_matrix = dijkstra(csgraph=csr_matrix(distances), directed=False, indices=0, return_predecessors=False)
    print(dist_matrix[-1])
    errors[n] = np.abs(dist_matrix[-1] - np.sqrt(2))
    
ax.semilogy(list(errors.keys()), list(errors.values()), "-o", label = "Regular grid")


ax.legend()
plt.show()