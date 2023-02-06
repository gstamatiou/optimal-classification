from wass_dist import wass_dist
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import construct_weightmatrix, graph_grad

# =============================================================================
# Test wass_dist for scalar measure being the difference of two diracs in 1 and 2D
# 1D test should output 1 and 2D test should output sqrt(2)
# =============================================================================

# 1D test
n = 50

X = np.linspace(0, 1, n)
W = construct_weightmatrix(X)

mu = np.zeros((n,1))
mu[0] = [1]
mu[-1] = [-1]

u1d = wass_dist(W, mu, np.zeros((n,1)), np.zeros((n,n,1)), 
                      tol = 1e-4, max_iter=1e5, verbose=False)

print((u1d * mu).sum())
plt.scatter(X, u1d, s = 1)

# 2D test 
n = 10*10
xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
X = np.array([xv.flatten(), yv.flatten()]).T

mu = np.zeros((n,1))
mu[0] = [1]
mu[-1] = [-1]
W = construct_weightmatrix(X)
u0 = np.random.rand(n,1)
u2d = wass_dist(W, mu, u0, -graph_grad(u0, W), tol = 1e-5, 
                max_iter=1e5, verbose=False)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], u2d, s = 2)
print((u2d * mu).sum())