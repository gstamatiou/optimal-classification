from ot_class import *
from scipy.optimize import NonlinearConstraint


n = 50 # Number of samples
k = 2
lab_n = 5 # Number of labels per class

# Generate data sets
X, labels = datasets.make_moons(n_samples = n, noise = 0.1, random_state=0) # Generate point clouds and labels
train_ind = gl.trainsets.generate(labels, rate = lab_n, seed = 0) # Generate indices of "labeled" nodes
train_labels = labels[train_ind]
W = gl.weightmatrix.knn(X,10).toarray()

mu = np.zeros((n,k))
mu[train_ind] = euclidean_basis(train_labels , k)- train_labels.sum(axis = 0)/train_ind.size

def f(u):
	n = W.shape[0]
	k = int(u.size/n)
	u = u.reshape((n,k))

	normed_gradu = np.apply_along_axis(np.linalg.norm, 2, graph_grad(u, W))
	
	return normed_gradu[np.tril_indices(n, k=-1)]

def cost(u, mu):
	u = u.reshape(mu.shape)
	return -(u * mu).sum()

nlc = NonlinearConstraint(f, -np.inf, np.ones(int(0.5*n*(n-1))))

res = minimize(cost, x0 = np.zeros(n * k), args=(mu), constraints = nlc)
print(res)
print(grad(res.x))