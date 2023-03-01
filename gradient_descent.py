import numpy as np
import graphlearning as gl
from sklearn import datasets
from numpy.linalg import norm
from helpers import euclidean_basis, save_models, load_models
from energy import jacobian
from ppoisson import ppoisson
import time

def grad_descent(gradient, start, rate, tolerance = 1e-2, max_steps = int(1e9)):
    x = start
    for _ in range(max_steps):
        diff = - rate * gradient(x)

        if norm(diff) < tolerance:
            return x
        x += diff

    return x

# Test gradient descent for x^2
# print(grad_descent(gradient=lambda v: 2 * v, start=-10.0, rate=0.2))

n = 100
k = 2
X, labels = datasets.make_moons(n_samples=n,noise=0.1, random_state = 0)
W = gl.weightmatrix.knn(X,10).toarray()
train_ind = gl.trainsets.generate(labels, rate=5)
train_labels = labels[train_ind]
one_hot_labels = euclidean_basis(train_labels, k)

models = load_models("twomoons_data")

p = 8

start_time = time.time()
u = grad_descent(lambda u: jacobian(u, W, train_ind, one_hot_labels, p), 
        start=models[4].u.flatten(), rate = 0.5, tolerance = 1e-2)
end_time = time.time()
runtime = (end_time - start_time)/60 # in minutes

print(u)
print(runtime)
#save_models(u, "grad.pickle")
