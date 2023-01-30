from ot_class import *

def jacobian(u_flattened, W, idx, y, p):
    n = W.shape[0]
    k = u_flattened.size//n
    m = y.shape[0]    
    
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    normed_gradu = np.apply_along_axis(np.linalg.norm, 2, gradu)
    y_bar = y.sum(axis = 0)/m

    A = W * normed_gradu **(p-2)
    d = A.sum(axis = 1)
    D = np.diag(d)
    L = D - A
    B = np.zeros( (n, k) )
    B[idx] = y - y_bar
    
    jac = np.matmul(L, u) - B

    return jac.flatten()

# Testing vectorized jacobian
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
    print(scipy.optimize.check_grad(penergy, new_jacobian, u.flatten(), W, train_ind, y, p))
