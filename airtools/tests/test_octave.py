#!/usr/bin/env python
import numpy as np
import pytest
from pytest import approx
from pathlib import Path
import airtools

Rmatlab = Path(__file__).resolve().parents[1] / "matlab"
"""
generate test problems from Julia by

using MatrixDepot
"""
x = np.array([1.0, 3.0, 0.5, 2.0])

used = ("identity", "fiedler")


@pytest.mark.parametrize("name", used)
def test_maxent(matrices, name):
    A = matrices
    oct2py = pytest.importorskip("oct2py")

    b = A @ x
    lamb = 2.5e-5

    with oct2py.Oct2Py(timeout=10, oned_as="column") as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.maxent(A, b, lamb).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.maxent(A, b, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.parametrize("name", used)
def test_kaczmarz(matrices, name):
    A = matrices
    oct2py = pytest.importorskip("oct2py")

    b = A @ x
    max_iter = 200
    lamb = 1.0
    x0 = np.zeros_like(x)

    with oct2py.Oct2Py(timeout=10, oned_as="column") as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.kaczmarz(A, b, max_iter, x0, {"lambda": lamb}).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.kaczmarz(A, b, x0=x0, max_iter=max_iter, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.parametrize("name", used)
def test_logmart(matrices, name):
    A = matrices
    oct2py = pytest.importorskip("oct2py")

    b = A @ x
    relax = 5.0
    max_iter = 2000
    sigma = 1.0

    with oct2py.Oct2Py(timeout=10, oned_as="column") as oc:
        oc.addpath(str(Rmatlab))
        x_matlab = oc.logmart(b, A, relax, [], sigma, max_iter).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.logmart(A, b, relax=relax, sigma=sigma, max_iter=max_iter)[0]
    assert x_est == approx(x_matlab, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
