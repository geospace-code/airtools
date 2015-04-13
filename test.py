#!/usr/bin/env python
"""selftest"""
from numpy import array
from numpy.testing import assert_array_almost_equal, assert_almost_equal
#%% kaczmarz
from kaczmarz import kaczmarz_ART
A = array([[1, 2, 0],[0, 4, 3]],dtype=float)
b = array([8,18],dtype=float)
x = kaczmarz_ART(A,b,50)[0]
assert_array_almost_equal(x,[ 0.91803279,  3.54098361,  1.27868852])
#%% maxent
from maxent import maxent
A = array([[1, 2, 0],[0, 4, 3]])
b = array([8,18])
x,rho,eta = maxent(A,b,1)
assert_array_almost_equal(x,[0.552883833066741, 3.621706597812032, 1.109718756265391])
assert_almost_equal(rho,0.274512808306942)
assert_almost_equal(eta,4.448824493430995)