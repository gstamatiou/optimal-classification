import numpy as np
from numpy.linalg import norm
from numpy.random import rand
import matplotlib.pyplot as plt
import graphlearning as gl
from scipy.optimize import minimize, LinearConstraint
import sklearn.datasets as datasets
import scipy, time, pickle
from pyvis.network import Network
from PIL import Image

def save_models(models, filename):
    with open('data_dumps/' + filename, 'wb') as f:
        pickle.dump(models, f)
        f.close()
    
    return

def load_models(filename):
    with open('data_dumps/' + filename, "rb") as f:
        models = pickle.load(f)
        f.close()
    
    return models

# Calculates vector of degrees for the weight matrix W
# W: (n,k) numpy array
def degrees(W):
    return W.sum(axis = 1)

# Example
# Calculates degrees of equilateral triangle
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# print(degrees(W))

def graph_div(V, W):
    n = V.shape[0]
    k = V.shape[2]
    div = ( V * (W[:, :, np.newaxis] + np.zeros((n,n,k))) ).sum(axis = 1)
    return div

def proxF(V, sigma):
    norms = norm(V, axis = 2)

    Z = np.copy(V)
    Z[norms <= sigma] = 0
    Z[norms > sigma] = ((norms[norms > sigma] - sigma)/norms[norms > sigma])[:, np.newaxis] * Z[norms > sigma]
    
    return Z

def proxG(u, tau, mu):
    return tau * mu + u

def power_iteration(K, x_0, termination_threshold = 1e-4, num_iterations = 10000000):
    b = x_0/norm(x_0)
    for _ in range(num_iterations):
        bb = K(b)/norm(K(b))

        if norm(b - bb) < termination_threshold:
            return bb, norm(K(bb))/norm(bb)
        b = bb

    return bb, norm(K(u))/norm(u)

def Kstar_K(u, W):
    return 2 * np.matmul((np.diag(W.sum(axis = 0)) - W), u)

def wass_dist(W, mu, u0, V0, theta = 1, threshold = 1e-1):
    difference = threshold+1
    
    n = W.shape[0]

    K = graph_grad
    Kstar = lambda V, W: -graph_div(V, W)

    
    _, eig = power_iteration(lambda u: Kstar_K(u, W), rand(n))
    K_norm = np.sqrt(eig)

    sigma = 1/ (2* K_norm)
    tau = 1/(2 * K_norm )

    u = u0
    u_bar = np.copy(u0)
    V = V0

    while difference > threshold:
        VV = proxF(V + sigma * K(u_bar, W), sigma)
        uu = proxG(u - tau * Kstar(VV, W), tau, mu)
        uu_bar = uu + theta * (uu - u)

        difference = np.linalg.norm(uu - u)
        V = VV
        u = uu
        u_bar = uu_bar
        

    return u

"""
Computes the weighted gradient of u
u: (n,k) matrix
W: (n,n) matrix

Returns:
gradu: (n, n, k)

where gradu[i, j, l] = W[i,j] * u[j,l]-u[i,l]
"""
def graph_grad(u, W):
    gradu = W[:, :, np.newaxis] * (-u[:, np.newaxis] + u)

    return gradu

"""
Computes the non-weighted gradient of u
u: (n,k) matrix

Returns:
gradu: (n, n, k)

where gradu[i, j, l] = u[j,l]-u[i,l]
"""

def grad(u):
    gradu = -u[:, np.newaxis] + u

    return gradu

# # Example:
# u = np.arange(6).reshape(3,2)
# print("u is\n", u)
# print("\nAnd gradu is")
# for i in range(u.shape[0]):
#     for j in range(u.shape[1]):
#         print(graph_grad(u)[i,j], end = "\t\t")
#     print()

"""
u_flattened: n*k matrix
W: weight matrix
y: (m,k) matrix
idx = vector of labelled indices
p = p constant for laplace operator
"""
def penergy(u_flattened, W, idx, y, p):
    k = y.shape[1]
    n = int(u_flattened.size/k)
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    y_bar = (1/y.shape[0]) * y.sum(axis = 0)
    
    first_summand = (1/(2*p)) * (W * (np.apply_along_axis(np.linalg.norm, 2, gradu) ** p)).sum()
    second_summand = np.sum( (y - y_bar) * u[idx] )

    return first_summand - second_summand

# ## Example (see pdf)
# u = np.array([[1,0], [0,1], [1/2, 1/2]])
# print(graph_grad(u))
# y = np.array([[1,0], [0,1]])
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# idx = [0, 1]
# p = 2

# #print(np.apply_along_axis(np.linalg.norm, 2, graph_grad(u)))
# print("2-energy is", penergy(u, W, idx, y, 2))
# print("3-energy is", penergy(u, W, idx, y, 3))

def jacobian(u_flattened, W, idx, y, p):
    n = W.shape[0]
    k = u_flattened.size//n
    m = y.shape[0]    
    
    u = u_flattened.reshape((n,k))
    gradu = grad(u)
    normed_gradu = np.apply_along_axis(np.linalg.norm, 2, gradu)
    y_bar = y.sum(axis = 0)/m
    
    if p > 2:
        # ... = w*normed_gradu**(p-2)
        a1 = W * (normed_gradu**(p-2)) + np.zeros((k, n, n)) # a1[s,i,r] = ...[i, r]
        A = np.transpose(a1, (1, 2, 0)) # A[i,r, s] = ...[i, r]

        u1 = np.zeros( (n, n, k) ) + u #u1[i, r, s] = u[r, s]

        u21 = np.zeros( (n,n,k) ) + u # u21[r, i, s] = u[i, s]
        u2 = np.swapaxes(u21, 0, 1) # u2[i, r, s] = u[i, s]

        C = u1 - u2

        B = np.zeros( (n, k) )
        B[idx] = y - y_bar

        jac = (A * C).sum(axis = 0) - B
    
    elif p == 2:
        a1 = W + np.zeros((k, n, n)) # a1[s,i,r] = W[i, r]
        A = np.transpose(a1, (1, 2, 0)) # A[i, r, s] = W[i, r]
        
        u1 = np.zeros( (n, n, k) ) + u #u1[i, r, s] = u[r, s]

        u21 = np.zeros( (n,n,k) ) + u # u2[r, i, s] = u[i, s]
        u2 = np.swapaxes(u21, 0, 1)

        C = u1 - u2

        B = np.zeros( (n, k) )
        B[idx] = y - y_bar
            
        jac = (A * C).sum(axis = 0) - B
    
    return jac.flatten()
    
# # Testing vectorized jacobian
# p = 6
# n = 15
# k = 2

# X,labels = datasets.make_moons(n_samples=n,noise=0.1)
# W = gl.weightmatrix.knn(X,10).toarray()
# train_ind = gl.trainsets.generate(labels, rate=5)
# train_labels = labels[train_ind]
# m = train_ind.size

# y = np.zeros((m, k))
# for i in range(train_ind.size):
#     y[i] = euclidean_basis(train_labels[i], k)
    

# u = np.random.random(size = (n,k))

# scipy.optimize.check_grad(penergy, jacobian, u.flatten(), W, train_ind, y, p)

"""
Returns the ith vector of the usual basis of R^k
Index starts at 0
"""
def euclidean_basis(i, k):
    eye = np.eye(k)
    return eye[i]

# Example
# print(euclidean_basis(0, 4))
# print(euclidean_basis(2, 4))

# Returns the labels that u predicts
def predict(u):
    return np.argmax(u, axis = 1)

# ############## Toy example using gradient descent
# y = np.array([[1,0], [0,1]])
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# idx = [0, 1]
# p = 2

# u = gradient_ppoisson(W, idx, y, p)
# print("Minimizer:\n", u)
# print("Energy of minimizer:", penergy(u, W, idx, y, p))
# print("Negative Laplacian values:", 2*u[0]-u[1]-u[2], 2*u[1]-u[0]-u[2], 2*u[2]-u[0]-u[1], sep="\n")
# print("Labels:", predict(u))

# ############## Toy example using GraphLearn
# y = np.array([[1,0], [0,1]])
# W = np.array([[0, 1, 1], [1,0,1], [1,1,0]])
# d = degrees(W)
# idx = [0, 1]
# n = 3
# k = 2
# p = 2

# model = gl.ssl.poisson(W)
# u = model.fit(idx, np.array([0, 1]))
# # print(2*u[0]-u[1]-u[2])
# # print(2*u[1]-u[0]-u[2])
# # print(2*u[2]-u[0]-u[1])
# # print(u[0] + u[1] + u[2])
# print("GraphLearn solution: \n", u)

# # Toy example using custom implementation
# my_u = gradient_ppoisson(W, idx, y, p)
# print("Custom solution: \n", my_u)

def construct_data(n, k = 2, labels_per_class=5):
    # Generate training data and label sets
    X,labels = datasets.make_moons(n_samples=n,noise=0.1, random_state = 0)
    W = gl.weightmatrix.knn(X,10).toarray()

    train_ind = gl.trainsets.generate(labels, rate=labels_per_class, seed = 0)
    train_labels = labels[train_ind]
    m = train_ind.size

    # Construction of measure mu (onehot encoded labels)
    # mu[i] = [0, 0] if i is  not in the labeld dataset
    # mu[i] = [1,0] - (mean of train_labels) if i is in the labeled dataset and its label is 1
    # mu[i] = [0,1] - (mean of train_labels) if i is in the labeled dataset and its label is 0
    
    mean = (labels_per_class/m) * np.ones(k)

    a_idx = train_ind[np.argwhere(labels[train_ind] == 1)].flatten()
    b_idx = train_ind[np.argwhere(labels[train_ind] == 0)].flatten()
    mu = np.zeros((n,k))
    mu[a_idx] = [1,0] - mean
    mu[b_idx] = [0,1] - mean

    return X, labels, train_ind, W, mu


# train_labels should be one-hot encoded
def construct_mu(n, k, train_ind, train_labels):
    mu = np.zeros((n,k))
    
    mu[train_ind] = train_labels - train_labels.sum(axis = 0)/train_ind.size

    return mu

class ppoisson():
    def __init__(self, p, W):
        self.p = p
        self.W = W
        self.n = W.shape[0]
        self.u = None
        self.fitted = False
        self.predicted = False
    
    # train_ind: Indices of (few) labeled points
    # train_labels: labels of labaled points (one hot encoding)
    # start: starting point (n,k) array
    def fit(self, train_ind, train_labels, start = np.zeros(1)):
        if self.fitted:
            return self.u
        
        start_time = time.time()

        self.k = train_labels.shape[1]
        d = degrees(self.W)
        eye = np.eye(self.k)


        if np.count_nonzero(start) == 0:
            model = gl.ssl.poisson(self.W, solver='gradient_descent')
            integer_coded_train_labels = np.argmax(train_labels, axis = 1)

            start = model.fit(train_ind, integer_coded_train_labels) # model's fit 
                                                                     # doesn't expect one hot encoding
        

        constrain_matrix = np.concatenate([d[i] * eye for i in range(self.n)], axis = 1)
        linear_constraint = LinearConstraint(constrain_matrix, np.zeros(self.k), np.zeros(self.k))
        
        start = start.flatten()
        res = minimize(penergy, x0 = start, args = (self.W, train_ind, train_labels, self.p), 
                jac = jacobian, method = 'trust-constr', constraints = linear_constraint)
        
        self.u = res.x.reshape(self.n,self.k)
        self.fitted = True
        
        end_time = time.time()
        self.runtime = (end_time - start_time)/60 # in minutes

        return self.u
        
    def predict(self):
        if not self.fitted:
            print("Not fitted yet")
            return -1
        if self.predicted:
            return self.predictions
        
        self.predictions = np.argmax(self.u, axis = 1)
        self.predicted = True

        return self.predictions

    
    def fit_predict(self, train_ind, train_labels, start = np.zeros(1)):
        self.fit(train_ind, train_labels, start)
        
        return self.predict()

    # labels: integer valued
    def accuracy(self, labels):
        if not self.predicted:
            self.predictions = self.predict()
        
        return 1 - np.count_nonzero(labels - self.predictions)/self.n

    def print_info():
        info_str = f"########### Gradient Descent (w/ Jacobian) for p = {self.p}\n"\
                        f"\nAccuracy = {self.accuracy() * 100:.2f}%\n"\
                        f"Runtime = {self.runtime:.2f} min"

        print(info_str)

def create_graph(X, graph_size = 100, file_name = "example.html", max_width = 40):
    resized_X = np.transpose(X[:graph_size].reshape(X[:graph_size].shape[0], 28, 28), (0, 2, 1))
    reweighted_W = W[:graph_size, :graph_size]
    reweighted_W = max_width * (1/reweighted_W.max()) * reweighted_W

    net = Network()

    for i in range(resized_X.shape[0]):
        img = Image.fromarray(resized_X[i] * 255)
        img = img.convert('RGB')
        img.save(f"data/{i}.png")
        net.add_node(i, shape = 'image', image = f'data/{i}.png')

    for i in range(resized_X.shape[0]):
        for j in range(resized_X.shape[0]):
            if i != j and W[i,j] > 0:
                net.add_edge(i, j, width = reweighted_W[i,j])

    net.toggle_physics(False)
    net.show(file_name)
    
    return 0

