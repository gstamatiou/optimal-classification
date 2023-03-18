import tensorflow as tf
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

mu = tf.scatter_nd(tf.constant([[0],[n-1]]), tf.constant([[1],[-1]], dtype="float32"), shape=[n,1])
# mu = np.zeros((n,1))
# mu[0] = [1]
# mu[-1] = [-1]

u1d = wass_dist(tf.constant(W, dtype="float32"), mu, tf.zeros((n,1)), tf.zeros((n,n,1)), 
                      tol = 1e-4, max_iter=1e5, verbose=False)

print(tf.reduce_sum(u1d * mu))
plt.scatter(X, u1d, s = 1)

# 2D test 
n = 10*10
xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(n))), np.linspace(0, 1, int(np.sqrt(n))))
X = np.array([xv.flatten(), yv.flatten()]).T

# mu = np.zeros((n,1))
# mu[0] = [1]
# mu[-1] = [-1]
mu = tf.scatter_nd(tf.constant([[0],[n-1]]), tf.constant([[1],[-1]], dtype="float32"), shape=[n,1])

W = construct_weightmatrix(X)
u0 = tf.random.uniform([n,1])
u2d = wass_dist(tf.constant(W, dtype="float32"), mu, u0, tf.zeros([n,n,1]), tol = 1e-5, 
                max_iter=1e5, verbose=False)
    

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], u2d, s = 2)
print(tf.reduce_sum(u2d * mu))
