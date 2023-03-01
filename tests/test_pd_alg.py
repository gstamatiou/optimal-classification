import numpy as np
from numpy.random import rand, seed
from pd_alg import power_iteration, pd_alg
from scipy.optimize import minimize

def proxF(u, sigma):
	v = u
	u_normed = np.apply_along_axis(np.linalg.norm, 1, v)
	v[u_normed >= 1] = np.diag(1/u_normed[u_normed >= 1]) @ v[u_normed >= 1]

	return v

def proxG(u, tau):
	return 1/(2 * tau + 1) * (u + 2 * tau * u0)

def K(x):
	return x

def Kstar(x):
	return x

def energy(u):
	u = u.reshape((n,k))

	F = np.apply_along_axis(np.linalg.norm, 1, u).sum()
	G = (np.apply_along_axis(np.linalg.norm, 1, u - u0)**2).sum()

	return F+G

n = 10	
k = 3
seed(0)
u0 = rand(n,k) * 10 - 5

Kstar_K = lambda x: Kstar(K(x))

_, eig = power_iteration(Kstar_K, rand(n,k))
K_norm = np.sqrt(eig)

sigma = 1/(2 * K_norm)
tau = 1/(2 * K_norm)

pd_x = pd_alg(lambda x: proxF(x, sigma), lambda y: proxG(y, tau), K, Kstar, tau, sigma, rand(n,k), rand(n,k), tol = 1e-15)
min_x = minimize(energy, rand(n*k))
print(pd_x - min_x.x.reshape((n,k)))
print(energy(pd_x.flatten()) - energy(min_x.x))



# # Test proxF
# n = 10	
# k = 3
# seed(0)
# u0 = rand(n,k) * 10 - 5

# v = np.zeros((n,k))
# for i in range(n):
# 	if norm(u0[i]) >= 1:
# 		v[i] = 1/norm(u0[i]) * u0[i]
# 	else:
# 		v[i] = u[i]

# print(v - proxF(u0, 0))
