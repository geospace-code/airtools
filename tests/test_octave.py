#!/usr/bin/env python
"""
Tests Python code vs. Matlab by original luminiferous authors
...using Octave, of course
"""
from numpy import array
from numpy.testing import run_module_suite, assert_array_almost_equal,assert_almost_equal
from oct2py import Oct2Py
"""
generate test problems from Julia by

using MatrixDepot
matrixdepot("deriv2",3,false)
"""
A = array([[-0.0277778, -0.0277778, -0.00925926],
           [-0.0277778, -0.0648148, -0.0277778],
           [-0.00925926,-0.0277778, -0.0277778 ]])
b = array([-0.01514653483985129,
            -0.03474793286789414,
            -0.022274315940957783])
x_true = array([0.09622504486493762,
                0.28867513459481287,
                0.48112522432468807])

oc = Oct2Py(timeout=10,convert_to_float=True,oned_as='column')
oc.addpath('../matlab')

def test_maxent():
#%% first with Python   
    from airtools.maxent import maxent
               
    x_python,rho,eta = maxent(A,b,0.00002)
    assert_array_almost_equal(x_python,x_true)

#%% then with Octave using original Matlab code
    
    x_matlab =oc.maxent(A,b,0.00002).squeeze()   
    assert_array_almost_equal(x_matlab,x_true)

def test_kaczmarz():
    from airtools.kaczmarz import kaczmarz
 
    x_python = kaczmarz(A,b,200)[0]
    assert_array_almost_equal(x_python,x_true)
    
    x_matlab = oc.kaczmarz(A,b,200).squeeze()
    assert_array_almost_equal(x_matlab,x_true)
    
if __name__ == '__main__':
    run_module_suite()
