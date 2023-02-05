from wass_dist import wass_dist
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import construct_weightmatrix, graph_grad

# Should calculate the wasserstein distance betwwen pts 0, 1 in R
n = 50

X = np.linspace(0, 1, n)
W = construct_weightmatrix(X)

mu = np.zeros((n,1))
mu[0] = [1]
mu[-1] = [-1]

u1d = wass_dist(W, mu, np.zeros((n,1)), np.zeros((n,n,1)), 
                      tol = 1e-4, max_iterations=1e5, verbose=False)

print((u1d * mu).sum())
plt.scatter(X, u1d, s = 1)
plt.savefig("figures/Wasserstein_potential_1d.png")
#Should output 1, i.e. the eucl. dist between X[0], X[-1]



n = 20*20

xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
X = np.array([xv.flatten(), yv.flatten()]).T

mu = np.zeros((n,1))
mu[0] = [1]
mu[-1] = [-1]
W = construct_weightmatrix(X)
u0 = np.random.rand(n,1)
u2d = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-3, 
                max_iterations=1e5, verbose=False)

#plt.scatter(X[:, 0], X[:, 1], c = u2d, s = 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], u2d)
fig.savefig("figures/Wasserstein_potential_2d.png")
print((u2d * mu).sum())