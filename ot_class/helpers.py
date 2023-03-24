import numpy as np
import pickle
import graphlearning as gl
import sklearn.datasets as datasets
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


"""
Returns the ith vector of the usual basis of R^k
Index starts at 0
"""
def euclidean_basis(i, k):
    eye = np.eye(k)
    return eye[i]

"""
Returns the accuracy (in [0,1]) of the predictions
predictions: (n,) array with elements from 0, ..., k
labels: (n,) array with elements from 0, ..., k
"""
def calculate_accuracy(predictions, labels):
    return 1 - np.count_nonzero(predictions - labels)/labels.size

# Example
# print(euclidean_basis(0, 4))
# print(euclidean_basis(2, 4))

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

