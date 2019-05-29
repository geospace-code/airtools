from __future__ import division
import logging
from numpy import zeros, unique, asarray
from numpy.linalg import norm
from scipy.sparse import issparse
'''
Michael Hirsch May 2014

 tested with Dense and Sparse arrays. (Aug 2014)

 inputs:
 A:  M x N 2-D projection matrix
 b:  N x 1 1-D vector of observations
 maxIter: maximum number of ART iterations
 x0: N x 1 1-D vector of initialization (a guess at x)
 lamb: relaxation parameter (see Herman Ch.11.2)
 stopmode: {None, MDP}  stop before maxIter if solution is good enough
             (MDP is Morozov Discrepancy Principle)
 nonneg: enforces non-negativity of solution

 outputs:
 x: the estimated solution of A x = b
 residual: the error b-Ax

 References:
 Herman, G. " Fundamentals of Computerized Tomography", 2nd Ed., Springer, 2009
 Natterer, F. "The mathematics of computed tomography", SIAM, 2001
'''


def kaczmarz(A, b, maxIter=8, x0=None, lamb=1, stopmode=None, taudelta=0, nonneg=True):
    assert 0. <= lamb <= 2., 'unstable relaxation parameter'
# %% user parameters
    residual = None  # init

    n = A.shape[1]  # only need rows

    if x0 is None:  # we'll use zeros
        x0 = zeros(n, order='F')  # 1-D vector

    if stopmode is None or stopmode.lower() == 'iter':  # just use number of iterations
        sr = 0
    elif stopmode.lower() == 'mdp' or stopmode.lower() == 'dp':
        sr = 1
        if taudelta == 0:
            logging.warning('you used tauDelta=0, which effectively disables Morozov discrepancy principle')
    else:
        sr = 0
        logging.error("didn't understand stopmode command, defaulted to maximum iterations")

# %% disregard all-zero columns of A
    if issparse(A):
        A = A.tocsr()  # save time if it was csc sparse
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = asarray(A.multiply(A).sum(axis=1)).squeeze()  # 50 times faster than dense for 1024 x 100000 A
    else:  # is dense A
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = norm(A, ord=2, axis=1)**2  # timeit same for norm() and A**2.sum(axis=1)

    goodRows = unique(A.nonzero()[0])

    x = x0.copy()  # we'll leave the original x0 alone, and make a copy in x
    iIter = 0
    stop = False  # will always run at least once
    while not stop:  # for each iteration
        for iRow in goodRows:  # only not all-zero rows
            # denominator AND numerator are scalar!
            # den = np.linalg.norm(A[iRow,:],2)**2
            # print(RowNormSq[iRow] == den)
            # num = ( b[iRow] - A[iRow,:] @ x )
            # x = x + (lamb * num/den) @ A[iRow,:]
            x += lamb * (b[iRow] - A[iRow, :] @ x) / RowNormSq[iRow] * A[iRow, :]  # first two terms are scalar always

            if nonneg:
                x[x < 0] = 0

        iIter += 1
        # handle stop rule
        stop = iIter > maxIter
        if sr == 0:  # no stopping till iterations are done
            pass
        elif sr == 1:
            residual = b - A @ x
            residualNorm = norm(residual, 2)
            stop |= (residualNorm <= taudelta)
        if iIter % 200 == 0:  # print update every N loop iterations for user comfort
            residualNorm = norm(b - A @ x, 2)  # NOT a duplicate for sr==0 !
            print('Iteration {},  ||residual|| = {:.2f}'.format(iIter, residualNorm))

    return x, residual, iIter-1
