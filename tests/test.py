#!/usr/bin/env python
"""selftest"""
from numpy.linalg import svd
from numpy import array,mat,squeeze
from cvxopt import matrix
from scipy import sparse
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_allclose,run_module_suite
#
import airtools.lsqlin as lsqlin

def test_kaczmarz():
    from airtools.kaczmarz import kaczmarz_ART
    A = array([[1, 2, 0],[0, 4, 3]])
    b = array([8,18])
    x = kaczmarz_ART(A,b,50)[0]
    assert_array_almost_equal(x,[ 0.91803279,  3.54098361,  1.27868852])

def test_maxent():
    from airtools.maxent import maxent
    A = [[1, 2, 0],[0, 4, 3]]
    b = [8,18]
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
    eta = picard(U,s,V)[0]
    assert_array_almost_equal(eta,[ 0.02132175, 0.00238076, 0.04433971])

def test_lsqlin():
    # simple Testing routines
    C = array(mat('''0.9501,0.7620,0.6153,0.4057;
    0.2311,0.4564,0.7919,0.9354;
    0.6068,0.0185,0.9218,0.9169;
    0.4859,0.8214,0.7382,0.4102;
    0.8912,0.4447,0.1762,0.8936'''))
    sC = sparse.coo_matrix(C)
    csC = lsqlin.scipy_sparse_to_spmatrix(sC)

    A = array(mat('''0.2027,0.2721,0.7467,0.4659;
    0.1987,0.1988,0.4450,0.4186;
    0.6037,0.0152,0.9318,0.8462'''))
    sA = sparse.coo_matrix(A)
    csA = lsqlin.scipy_sparse_to_spmatrix(sA)

    d = array([0.0578, 0.3528, 0.8131, 0.0098, 0.1388])
    md = matrix(d)

    b =  array([0.5251, 0.2026, 0.6721])
    mb = matrix(b)

    lb = array([-0.1] * 4)
    mlb = matrix(lb)
    mmlb = -0.1

    ub = array([2] * 4)
    mub = matrix(ub)
    mmub = 2


    opts = {'show_progress': False}

    for iC in [C, sC, csC]:
        for iA in [A, sA, csA]:
            for iD in [d, md]:
                for ilb in [lb, mlb, mmlb]:
                    for iub in [ub, mub, mmub]:
                        for ib in [b, mb]:
                            ret = lsqlin.lsqlin(iC, iD, 0, iA, ib, None, None, ilb, iub, None, opts)
                            assert_allclose(squeeze(ret['x']),[-1.00e-01, -1.00e-01,  2.15e-01,  3.50e-01],rtol=1e-2)


    #test lsqnonneg
    C = array([[0.0372, 0.2869], [0.6861, 0.7071], [0.6233, 0.6245], [0.6344, 0.6170]]);
    d = array([0.8587, 0.1781, 0.0747, 0.8405]);
    ret = lsqlin.lsqnonneg(C, d, {'show_progress': False})
    assert_allclose(squeeze(ret['x']),[2.5e-7,6.93e-1],rtol=1e-2)

if __name__ == '__main__':
    run_module_suite()
