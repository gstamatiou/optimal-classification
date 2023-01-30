from ot_class import *

def slow_penergy(u_flattened, W, idx, y, p):
    k = y.shape[1]
    n = W.shape[0]
    m = y.shape[0]
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    normed_gradu = np.apply_along_axis(np.linalg.norm, 2, gradu)
    y_bar = (1/y.shape[0]) * y.sum(axis = 0)
    
    first_summand = 0
    for i in range(n):
        for j in range(n):
            first_summand += W[i, j] * normed_gradu[i,j] ** p
    first_summand = first_summand / (2 * p)

    second_summand = np.sum( (y - y_bar) * u[idx] )

    return first_summand - second_summand

n = 100
k = 2

X,labels = datasets.make_moons(n_samples=n,noise=0.1)
W = gl.weightmatrix.knn(X,10).toarray()
train_ind = gl.trainsets.generate(labels, rate=5)
train_labels = labels[train_ind]
m = train_ind.size


y = np.zeros((m, k))
for i in range(train_ind.size):
    y[i] = euclidean_basis(train_labels[i], k)
    

u = np.random.random(size = (n,k))

for p in [10, 50, 100, 200, 300]:
    print(penergy(u.flatten(), W, train_ind, y, p) - slow_penergy(u.flatten(), W, train_ind, y, p))

