#!/usr/bin/env python
"""selftest"""
from numpy.linalg import svd
from numpy import array
from numpy.testing import assert_array_almost_equal, assert_almost_equal

def test_kaczmarz():
    from airtools.kaczmarz import kaczmarz_ART
    A = array([[1, 2, 0],[0, 4, 3]],dtype=float)
    b = array([8,18],dtype=float)
    x = kaczmarz_ART(A,b,50)[0]
    assert_array_almost_equal(x,[ 0.91803279,  3.54098361,  1.27868852])

def test_maxent():
    from airtools.maxent import maxent
    A = array([[1, 2, 0],[0, 4, 3]])
    b = array([8,18])
    x,rho,eta = maxent(A,b,1)
    assert_array_almost_equal(x,[0.552883833066741, 3.621706597812032, 1.109718756265391])
    assert_almost_equal(rho,0.274512808306942)
    assert_almost_equal(eta,4.448824493430995)

def test_rzr():
    from airtools.rzr import rzr
    A = array([[1,2,3],
               [0,0,0],
               [4,5,6]])
    b = array([1,2,3])
    Ar,br,g = rzr(A,b)
    assert_array_almost_equal(Ar,array([[1,2,3],
                                        [4,5,6]]))
    assert_array_almost_equal(br,array([1,3]))   

def test_picard():
    from airtools.picard import picard
    U,s,V = svd(array([[3,2,2],
                       [2,3,-2],
                       [2,3,4]]))
    eta = picard(U,s,V)
    assert_array_almost_equal(eta,[ 0.02132175, 0.00238076, 0.04433971])
    
if __name__ == '__main__':
    test_kaczmarz()
    test_maxent()
    test_rzr()
    test_picard()