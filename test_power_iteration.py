from ot_class import *

np.random.seed(0)
x1 = np.random.rand(2) * 2
np.random.seed(1)
x2 = np.random.rand(2) * 2
mu = np.array([[0.5, -0.5]])
W = np.array([[0, norm(x1 - x2)], [norm(x1 - x2), 0]])

def f(u):
    return -graph_div(graph_grad(u,W), W)

_, normed_squared = power_iteration(f, rand(2))
print(np.sqrt(normed_squared) == np.sqrt(2 * W[0,1] ** 2))

