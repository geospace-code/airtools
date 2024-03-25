from pathlib import Path

import numpy as np

import pytest
from pytest import approx

import airtools

try:
    from .matlab_engine import matlab_engine

    matlab_skip = False
except (ImportError, RuntimeError):
    matlab_skip = True


Rmatlab = Path(__file__).resolve().parents[3]
"""
generate test problems from Julia by

using MatrixDepot
"""
x = np.array([1.0, 3.0, 0.5, 2.0])

used = ("identity", "fiedler")


@pytest.mark.skipif(matlab_skip, reason="Matlab Engine not available")
@pytest.mark.parametrize("name", used)
def test_maxent(matrices, name):

    eng = matlab_engine()
    eng.addpath(eng.genpath(str(Rmatlab)), nargout=0)

    A = matrices
    b = A @ x
    lamb = 2.5e-5

    x_matlab = eng.airtools.maxent(A, b, lamb).squeeze()

    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.maxent(A, b, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.skipif(matlab_skip, reason="Matlab Engine not available")
@pytest.mark.parametrize("name", used)
def test_kaczmarz(matrices, name):

    eng = matlab_engine()
    eng.addpath(eng.genpath(str(Rmatlab)), nargout=0)

    A = matrices
    b = A @ x
    max_iter = 200
    lamb = 1.0
    x0 = np.zeros_like(x)

    x_matlab = eng.airtools.kaczmarz(A, b, max_iter, x0, {"lambda": lamb}).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.kaczmarz(A, b, x0=x0, max_iter=max_iter, lamb=lamb)[0]
    assert x_est == approx(x_matlab)


@pytest.mark.skipif(matlab_skip, reason="Matlab Engine not available")
@pytest.mark.parametrize("name", used)
def test_logmart(matrices, name):

    eng = matlab_engine()
    eng.addpath(eng.genpath(str(Rmatlab)), nargout=0)

    A = matrices

    b = A @ x
    relax = 5.0
    max_iter = 2000
    sigma = 1.0

    x_matlab = eng.airtools.logmart(b, A, relax, [], sigma, max_iter).squeeze()
    assert x_matlab == approx(x, rel=0.01)

    x_est = airtools.logmart(A, b, relax=relax, sigma=sigma, max_iter=max_iter)[0]
    assert x_est == approx(x_matlab, rel=1e-5)
