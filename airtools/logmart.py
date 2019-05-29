#!/usr/bin/env python
"""
solve b=Ax using parallel log-ent mart.
Original Matlab by Joshua Semeter
port to Python by Michael Hirsch
"""
import numpy as np


def logmart(A: np.ndarray, b: np.ndarray,
            *,
            relax: float = 1.,
            x0: float = None,
            sigma: float = None,
            max_iter: int = 200) -> tuple:
    """
    Displays delta Chisquare.
    Program is stopped if Chisquare increases.
    A is NxM array
    b is Nx1 vector
    returns Mx1 vector

    relax	     user specified relaxation constant	(default is 20.)
    x0	     user specified initial guess (N vector)  (default is backproject y, i.e., y#A)
    max_iter	user specified max number of iterations (default is 20)

    AUTHOR:	Joshua Semeter
    LAST MODIFIED:	5-2015

      Simple test problem
    A = np.diag([5, 5, 5])
    x = np.array([1,2,3])
    b = A @ x
    """
# %% parameter check
    if b.ndim != 1:
        raise ValueError('y must be a column vector')
    if A.ndim != 2:
        raise ValueError('A must be a matrix')
    if A.shape[0] != b.size:
        raise ValueError('A and y number of rows must match')
    if not isinstance(relax, float):
        raise ValueError('relax is a scalar float')

    b = b.copy()  # needed to avoid modifying outside this function!
# %% set defaults
    if sigma is None:
        sigma = np.ones_like(b)

    if x0 is None:  # backproject
        x = A.T @ b / A.ravel().sum()
        xA = A @ x
        x = x * b.max() / xA.max()
    elif isinstance(x0, (float, int)) or x0.size == 1:  # replicate
        x = x0 * np.ones_like(b)
    else:
        x = x0
# %% make sure there are no 0's in y
    b[b <= 1e-8] = 1e-8
    # W=sigma;
    # W=linspace(1,0,size(A,1))';
    # W=rand(size(A,1),1);
    W = np.ones(A.shape[0])
    W = W / W.sum()

    i = 0
    done = False
    arg = ((A @ x - b)/sigma)**2.
    chi2 = np.sqrt(arg.sum())

# %%  iterate solution, plot estimated data (diag elems of x#A)
    while not done:
        i += 1
        xold = x
        xA = A @ x
        t = (1/xA).min()
        C = relax*t*(1.-(xA/b))
        x = x / (1 - x*(A.T @ (W*C)))
# %% monitor solution
        chiold = chi2
        chi2 = np.sqrt((((xA - b)/sigma)**2).sum())
        # dchi2=(chi2-chiold);
        done = ((chi2 > chiold) & (i > 2)) | (i == max_iter) | (chi2 < 0.7)
# %% plot
#        figure(9); clf; hold off;
#        Nest=reshape(x,69,83);
#        imagesc(Nest); caxis([0,1e11]);
#        set(gca,'YDir','normal'); set(gca,'XDir','normal');
#        pause(0.02)
    y_est = A @ xold

    return xold, y_est, chi2, i
