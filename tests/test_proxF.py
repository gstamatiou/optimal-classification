from ot_class import *
from numpy.random import seed, rand

def slow_proxF(V, sigma):
	n = V.shape[0]

	Z = np.zeros(V.shape)
	for i in range(n):
		for j in range(n):
			if norm(V[i,j]) > sigma:
				Z[i,j] = (norm(V[i,j]) - sigma)/norm(V[i,j]) * V[i,j]


	return Z

n = 10	
k = 3
seed(0)
V = rand(n,n,k) * 10 - 5

Kstar_K = lambda x: Kstar(K(x))

seed(1)
sigma = rand()


print(np.max(np.abs(slow_proxF(V, sigma) - proxF(V, sigma))))