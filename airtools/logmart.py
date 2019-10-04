#!/usr/bin/env python
"""
solve b=Ax using parallel log-entropy MART  (De Pierro 1991)

Original Matlab by Joshua Semeter
port to Python by Michael Hirsch
"""
import numpy as np
import math


def logmart(A: np.ndarray, b: np.ndarray,
            *,
            relax: float = 1.,
            x0: float = None,
            sigma: float = 1.,
            max_iter: int = 20) -> tuple:
    """
    estimation halted based on chi**2 value
    A and b must be all NON-NEGATIVE!

    Parameters
    ----------
    A: numpy.ndarray
        NxM array "projection"
    b: numpy.ndarray
        N column vector "observation"

    Returns
    -------
    x_est: numpy.ndarray
        M column vector estimate of "true" x in b = A @ x

    Matlab logmart.m AUTHOR: Joshua Semeter 5-2015

    >>> A = np.diag([5, 5, 5])
    >>> x = np.array([1,2,3])
    >>> b = A @ x
    """
# %% parameter check
    if b.ndim != 1:
        raise ValueError('y must be a column vector')
    if A.ndim != 2:
        raise ValueError('A must be a matrix')
    if A.shape[0] != b.size:
        raise ValueError('A and b: number of rows must match')
    if not isinstance(relax, (int, float)):
        raise ValueError('relax is a scalar float')
    if (A < 0).any():
        raise ValueError('A must be all non-negative')
    if (b < 0).any():
        raise ValueError('b must be all non-negative')

    b = b.copy()  # needed to avoid modifying outside this function!
    # %% make sure there are no 0's in b
    b[b <= 1e-8] = 1e-8
# %% set defaults
    if x0 is None:  # backproject
        x = A.T @ b / A.sum()
        x *= b.max() / (A @ x).max()
    elif isinstance(x0, (float, int)) or x0.size == 1:  # replicate
        x = x0 * np.ones_like(b)
    else:
        x = x0
    if not x.size == A.shape[1]:
        raise ValueError('x0 must be scalar or match Ncolumns of A')

    x[x < 1e-8] = 1e-8
    # W=sigma;
    # W=linspace(1,0,size(A,1))';
    # W=rand(size(A,1),1);
    W = np.ones(A.shape[0])
    W = W / W.sum()

    chi2 = chi_squared(A, b, x, sigma)
# %%  iterate solution, plot estimated data (diag elems of x#A)
    for i in range(max_iter):
        x_prev = x
        xA = A @ x
        t = (1/xA).min()
        C = relax * t * (1 - xA/b)
        x /= (1 - x*(A.T @ (W*C)))
# %% monitor solution
        chiold = chi2
        chi2 = chi_squared(A, b, x, sigma)
        if i > 1 and chi2 >= chiold:
            break
        # if chi2 < 0.7:
        #    break

    return x_prev, chi2, i


def chi_squared(A: np.ndarray, b: np.ndarray, x: np.ndarray, sigma: float) -> float:
    return math.sqrt((((A @ x - b)/sigma)**2).sum())
