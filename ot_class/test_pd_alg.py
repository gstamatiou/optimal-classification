import numpy as np
import tensorflow as tf
from numpy.random import rand, seed
from pd_alg import power_iteration, pd_alg
from scipy.optimize import minimize

def numpy_proxF(u, sigma):
    v = u
    u_normed = np.apply_along_axis(np.linalg.norm, 1, v)
    v[u_normed >= 1] = np.diag(1/u_normed[u_normed >= 1]) @ v[u_normed >= 1]

    return v

def proxF(u, sigma):
    u_normed = tf.linalg.norm(u, axis = 1)
    
    v = tf.scatter_nd(tf.where(u_normed >= 1), tf.expand_dims(1/tf.gather_nd(u_normed, tf.where(u_normed >= 1)), axis = 1) * tf.gather_nd(u, tf.where(u_normed >= 1)), shape=u.shape)
    v = tf.tensor_scatter_nd_add(v, tf.where(u_normed < 1), tf.gather_nd(u, tf.where(u_normed<1)))
    
    return v

def proxG(u, tau):
    return 1/(2 * tau + 1) * (u + 2 * tau * u0)

def K(x):
    return x

def Kstar(x):
    return x

def energy(u):
    u = u.reshape((n,k))

    # F = tf.reduce_sum(tf.linalg.norm(u, axis = 1))
    # G = tf.reduce_sum(tf.linalg.norm(u-u0, axis = 1)**2)
    F = np.apply_along_axis(np.linalg.norm, 1, u).sum()
    G = (np.apply_along_axis(np.linalg.norm, 1, u - u0)**2).sum()

    return F+G

n = 10    
k = 3
tf.random.set_seed(0)
u0 = tf.random.uniform([n,k],minval=-5, maxval =5, dtype="float32")

Kstar_K = lambda x: Kstar(K(x))

_, eig = power_iteration(Kstar_K, tf.random.uniform([n,k], dtype="float32"))
K_norm = tf.sqrt(eig)

sigma = 1/(2 * K_norm)
tau = 1/(2 * K_norm)

pd_x = pd_alg(lambda x: proxF(x, sigma), lambda y: proxG(y, tau), K, Kstar, tau, sigma, tf.random.uniform([n,k]), tf.random.uniform([n,k]), tol = 1e-15)
numpy_pd_x = pd_alg(lambda x: proxF(x, sigma), lambda y: proxG(y, tau), K, Kstar, tau, sigma, rand(n,k), rand(n,k), tol = 1e-15)
min_x = minimize(energy, rand(n*k))
print(pd_x - min_x.x.reshape((n,k)))
print(pd_x - numpy_pd_x)
print(energy(pd_x.numpy().flatten()) - energy(min_x.x))



# =============================================================================
# # Test proxF
# n = 10
# k = 3
# tf.random.set_seed(0)
# u0 = tf.random.uniform(shape = [n,k], minval = -1, maxval = 2, dtype="float64")
# 
# v = np.zeros((n,k))
# for i in range(n):
#     if np.linalg.norm(u0[i].numpy()) >= 1:
#         v[i] = 1/np.linalg.norm(u0[i].numpy()) * u0.numpy()[i]
#     else:
#         v[i] = u0[i]
# 
# print(v - numpy_proxF(u0.numpy(), 0))
# print(numpy_proxF(u0.numpy(), 0) - proxF(u0, 0))
# 
# =============================================================================
