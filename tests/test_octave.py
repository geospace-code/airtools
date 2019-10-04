#!/usr/bin/env python
import numpy as np
import pytest
from pytest import approx
from pathlib import Path
import airtools

Rmatlab = Path(__file__).resolve().parents[1]/'matlab'
"""
generate test problems from Julia by

using MatrixDepot
"""
A = {"identity": np.diag([5., 5., 5., 5.]),
     "forsythe": np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [1.49012e-8, 0, 0, 0]]),
     "gravity": np.array([[4.0, 1.41421,  0.357771, 0.126491],
                          [1.41421, 4.0, 1.41421, 0.357771],
                          [0.357771, 1.41421, 4.0, 1.41421],
                          [0.126491, 0.357771, 1.41421, 4.0]]),
     "fiedler": np.array([[0, 1, 2, 3],
                          [1, 0, 1, 2],
                          [2, 1, 0, 1],
                          [3, 2, 1, 0]]),
     "hilbert": np.array([[1., 1/2, 1/3, 1/4],
                          [1/2, 1/3, 1/4, 1/5],
                          [1/3, 1/4, 1/5, 1/6],
                          [1/4, 1/5, 1/6, 1/7]])
     }


x = np.array([1.,
              3.,
              0.5,
              2.])

used = ("identity", "fiedler")


@pytest.mark.parametrize("A", [A[k] for k in used], ids=used)
def test_maxent(A):
    oct2py = pytest.importorskip('oct2py')

    b = A @ x
    lamb = 2.5e-5

    with oct2py.Oct2Py(timeout=10, oned_as='column') as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.maxent(A, b, lamb).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.maxent(A, b, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.parametrize("A", [A[k] for k in used], ids=used)
def test_kaczmarz(A):
    oct2py = pytest.importorskip('oct2py')

    b = A @ x
    max_iter = 200
    lamb = 1.
    x0 = np.zeros_like(x)

    with oct2py.Oct2Py(timeout=10, oned_as='column') as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.kaczmarz(A, b, max_iter, x0, {'lambda': lamb}).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.kaczmarz(A, b, x0=x0, max_iter=max_iter, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.parametrize("A", [A[k] for k in used], ids=used)
def test_logmart(A):
    oct2py = pytest.importorskip('oct2py')

    b = A @ x
    relax = 5.
    max_iter = 2000
    sigma = 1.

    with oct2py.Oct2Py(timeout=10, oned_as='column') as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.logmart(b, A, relax, [], sigma, max_iter).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.logmart(A, b, relax=relax, sigma=sigma, max_iter=max_iter)[0]
    assert x_est == approx(x_matlab, rel=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
