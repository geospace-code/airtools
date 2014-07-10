#!/usr/bin/env python3
import numpy as np
from maxent import maxent

A = np.array([[1,3,5],[3,2,6],[92,68,1]])
b = np.array([3,67,1])
lamb = 0.2

xhat = maxent(A,b,lamb)
print(xhat)

