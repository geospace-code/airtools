#!/usr/bin/env python
from numpy import array
import pytest
from pytest import approx

from airtools.maxent import maxent
from airtools.kaczmarz import kaczmarz
"""
generate test problems from Julia by

using MatrixDepot
matrixdepot("deriv2",3,false)
"""
A = array([[-0.0277778, -0.0277778, -0.00925926],
           [-0.0277778, -0.0648148, -0.0277778],
           [-0.00925926, -0.0277778, -0.0277778]])
b = array([-0.01514653483985129,
           -0.03474793286789414,
           -0.022274315940957783])
x_true = array([0.09622504486493762,
                0.28867513459481287,
                0.48112522432468807])


def test_maxent():
    # %% first with Python

    x_python, rho, eta = maxent(A, b, 0.00002)
    assert x_python == approx(x_true, rel=1e-4)

# %% then with Octave using original Matlab code
    oct2py = pytest.importorskip('oct2py')

    oc = oct2py.Oct2Py(timeout=10, oned_as='column')
    oc.addpath('../matlab')

    x_matlab = oc.maxent(A, b, 0.00002).squeeze()
    assert x_matlab == approx(x_true)


def test_kaczmarz():
    x_python = kaczmarz(A, b, 200)[0]
    assert x_python == approx(x_true, rel=1e-4)

    oct2py = pytest.importorskip('oct2py')

    oc = oct2py.Oct2Py(timeout=10, oned_as='column')
    oc.addpath('../matlab')

    x_matlab = oc.kaczmarz(A, b, 200).squeeze()
    assert x_matlab == approx(x_true)


if __name__ == '__main__':
    pytest.main(['-x', __file__])
