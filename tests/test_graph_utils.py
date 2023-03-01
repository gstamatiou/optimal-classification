#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
from numpy.random import rand
from graph_utils import construct_weightmatrix

n = 50
X = rand(n)
res = construct_weightmatrix(X)

for i in range(n):
    for j in range(n):
        if i == j and res[i,j] != 1:
            raise Exception("Mistake on 1D matrix check for construct matrix")
        elif i != j and 1/norm(X[i] - X[j]) != res[i,j]:
            raise Exception("Mistake on 1D matrix check for construct matrix")
print("construct_matrix test for 1D matrices passed")

n = 50
X = rand(n,n)
res = construct_weightmatrix(X)
for i in range(n):
    for j in range(n):
        if i == j and res[i,j] != 1:
            raise Exception("Mistake on 2D matrix check for construct matrix")
        elif i != j and 1/norm(X[i] - X[j]) != res[i,j]:
            raise Exception("Mistake on 2D matrix check for construct matrix")
print("construct_matrix test for 2D matrices passed")