#!/usr/bin/env python
from numpy import array
import pytest
from pytest import approx
from pathlib import Path

Rmatlab = Path(__file__).resolve().parents[1]/'matlab'
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
    oct2py = pytest.importorskip('oct2py')

    oc = oct2py.Oct2Py(timeout=10, oned_as='column')
    oc.addpath(str(Rmatlab))

    x_matlab = oc.maxent(A, b, 2.5e-5).squeeze()
    assert x_matlab == approx(x_true, rel=0.1)


def test_kaczmarz():
    oct2py = pytest.importorskip('oct2py')

    oc = oct2py.Oct2Py(timeout=10, oned_as='column')
    oc.addpath(str(Rmatlab))

    x_matlab = oc.kaczmarz(A, b, 200).squeeze()
    assert x_matlab == approx(x_true, rel=0.1)


@pytest.mark.xfail(reason='issue with original Matlab code')
def test_logmart():
    oct2py = pytest.importorskip('oct2py')

    oc = oct2py.Oct2Py(timeout=10, oned_as='column')
    oc.addpath(str(Rmatlab))

    x_matlab = oc.logmart(b, A)
    assert x_matlab == approx(x_true, rel=1e-4)


if __name__ == '__main__':
    pytest.main(['-x', __file__])
