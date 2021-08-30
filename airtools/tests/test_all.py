#!/usr/bin/env python
"""
generate test problems from Julia by:
https://github.com/JuliaMatrices/MatrixDepot.jl

using MatrixDepot
matrixdepot("hilb", 3)

ref: https://www.maths.manchester.ac.uk/~higham/narep/narep172.pdf
"""
import pytest
from pytest import approx
from numpy.linalg import svd
import numpy as np
from scipy import sparse

import airtools

x = np.array([1.0, 3.0, 0.5, 2.0])

used = ("identity", "fiedler")


@pytest.mark.parametrize("name", used)
def test_kaczmarz(matrices, name):
    A = matrices
    x_est = airtools.kaczmarz(A, A @ x, max_iter=100, lamb=1.0)[0]
    assert x_est == approx(x, rel=0.01)


@pytest.mark.parametrize("name", used)
def test_logmart(matrices, name):
    A = matrices
    x_est = airtools.logmart(A, A @ x, relax=5, max_iter=2000)[0]
    assert x_est == approx(x, rel=0.01)


@pytest.mark.parametrize("name", used)
def test_maxent(matrices, name):
    A = matrices
    x_est = airtools.maxent(A, A @ x, lamb=1e-6)[0]
    assert x_est == approx(x, rel=0.01)


def test_rzr():
    A = np.array([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
    b = np.array([1, 2, 3])
    Ar, br, g = airtools.rzr(A, b)
    assert Ar == approx(np.array([[1, 2, 3], [4, 5, 6]]))
    assert br == approx(np.array([1, 3]))


def test_picard():
    U, s, V = svd(np.array([[3, 2, 2], [2, 3, -2], [2, 3, 4]]))
    eta = airtools.picard(U, s, V)[0]

    assert eta == approx([0.02132175, 0.00238076, 0.04433971], rel=1e-4)


def test_lsqlin():
    pytest.importorskip("cvxopt")
    import airtools.lsqlin as lsqlin

    # simple Testing routines
    C = np.array(
        [
            [0.9501, 0.7620, 0.6153, 0.4057],
            [0.2311, 0.4564, 0.7919, 0.9354],
            [0.6068, 0.0185, 0.9218, 0.9169],
            [0.4859, 0.8214, 0.7382, 0.4102],
            [0.8912, 0.4447, 0.1762, 0.8936],
        ]
    )
    sC = sparse.coo_matrix(C)
    csC = lsqlin.scipy_sparse_to_spmatrix(sC)

    A = np.array(
        [
            [0.2027, 0.2721, 0.7467, 0.4659],
            [0.1987, 0.1988, 0.4450, 0.4186],
            [0.6037, 0.0152, 0.9318, 0.8462],
        ]
    )
    sA = sparse.coo_matrix(A)
    csA = lsqlin.scipy_sparse_to_spmatrix(sA)

    d = np.array([0.0578, 0.3528, 0.8131, 0.0098, 0.1388])
    md = d  # matrix(d)

    b = np.array([0.5251, 0.2026, 0.6721])
    mb = b  # matrix(b)

    lb = np.array([-0.1] * 4)
    mlb = lb  # matrix(lb)
    mmlb = -0.1

    ub = np.array([2] * 4)
    mub = ub  # matrix(ub)
    mmub = 2

    opts = {"show_progress": False}

    for iC in [C, sC, csC]:
        for iA in [A, sA, csA]:
            for iD in [d, md]:
                for ilb in [lb, mlb, mmlb]:
                    for iub in [ub, mub, mmub]:
                        for ib in [b, mb]:
                            ret = lsqlin.lsqlin(
                                iC, iD, 0, iA, ib, None, None, ilb, iub, None, opts
                            )
                            assert ret["x"] == approx(
                                [-1.00e-01, -1.00e-01, 2.15e-01, 3.50e-01], rel=1e-2
                            )

    # test lsqnonneg
    C = np.array([[0.0372, 0.2869], [0.6861, 0.7071], [0.6233, 0.6245], [0.6344, 0.6170]])
    d = np.array([0.8587, 0.1781, 0.0747, 0.8405])
    ret = lsqlin.lsqnonneg(C, d, {"show_progress": False})
    assert ret["x"] == approx([2.5e-7, 6.93e-1], rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
