from wass_dist import wass_dist
import numpy as np
import matplotlib.pyplot as plt
from graph_utils import construct_weightmatrix, graph_grad
import graphlearning as gl


# =============================================================================
# Test wass_dist for scalar measure being the difference of two diracs in 1 and 2D
# 1D test should output 1 and 2D test should output sqrt(2)
# =============================================================================

# 1D test
def epsilon_matrix(X, epsilon):
    normed_diff = np.apply_along_axis(np.linalg.norm, 2, X[:, np.newaxis] - X)
    W = np.zeros((X.shape[0], X.shape[0]))
    
    W[normed_diff <= epsilon] = 1/epsilon
    
    return W

n = 30

X = np.zeros((n,2))
X[:, 0] = np.linspace(0, 1, n)
W = gl.weightmatrix.epsilon_ball(X, 10, kernel='singular').toarray()

mu = np.zeros((n,1))
mu[0] = [1]
mu[-1] = [-1]

u1d = wass_dist(W, mu, np.zeros((n,1)), np.zeros((n,n,1)), 
                      tol = 1e-4, max_iter=1e5, verbose=False)

print((u1d * mu).sum())
plt.scatter(X[:, 0], u1d, s = 1)