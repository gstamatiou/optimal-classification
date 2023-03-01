import time
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
from graph_utils import degrees, predict
from sklearn import datasets
from helpers import euclidean_basis, load_models
from scipy.optimize import minimize, LinearConstraint
from ppoisson import ppoisson
from energy import jacobian, hessian, penergy
    

p = 8
n = 100
k = 2
X,labels = datasets.make_moons(n_samples=n,noise=0.1, random_state = 0)

W = gl.weightmatrix.knn(X,10).toarray()
train_ind = gl.trainsets.generate(labels, rate=5)
train_labels = labels[train_ind]
m = train_ind.size

y = np.zeros((m, k))
for i in range(train_ind.size):
    y[i] = euclidean_basis(train_labels[i], k)

d = degrees(W)
eye = np.eye(k)

constrain_matrix = np.concatenate([d[i] * eye for i in range(n)], axis = 1)
linear_constraint = LinearConstraint(constrain_matrix, np.zeros(k), np.zeros(k))

models = load_models("twomoons_data")

start_time = time.time()
res = minimize(penergy, x0 = models[255].u.flatten(), args = (W, train_ind, y, p),
               jac = jacobian, hess = hessian, method = 'trust-constr', constraints = linear_constraint)
end_time = time.time()

u = res.x.reshape((n,k))
plt.scatter(X[:, 0], X[:, 1], c = predict(u))

print(f"Runtime = {(end_time - start_time)/60:.2f} min")
