import logging
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse


def kaczmarz(A: np.ndarray,
             b: np.ndarray,
             *,
             max_iter: int = 8,
             x0: np.ndarray = None,
             lamb: float = 1.,
             stop_mdp: bool = False,
             taudelta: float = 0,
             nonneg: bool = True):
    '''
    Michael Hirsch May 2014

    tested with Dense and Sparse arrays. (Aug 2014)

    Parameters
    ----------
     A: numpy.ndarray
         M x N 2-D projection matrix
     b: numpy.ndarray
         N x 1 1-D vector of observations
     max_iter:  int
         maximum number of ART iterations
     x0: numpy.ndarray
         N x 1 1-D vector of initialization (a guess at x)
     lamb: float
         relaxation parameter (see Herman Ch.11.2)
     stop_mdp: bool
         stop before maxIter if solution is good enough (MDP is Morozov Discrepancy Principle)
    tau_delta: float
        stopping condition
     nonneg: bool
         enforces non-negativity of solution

    Results
    -------
    x: np.ndarray
        the estimated solution of A @ x = b
    residual: np.ndarray
        the error b-A@x

    References
    ----------
    Herman, G. " Fundamentals of Computerized Tomography", 2nd Ed., Springer, 2009
    Natterer, F. "The mathematics of computed tomography", SIAM, 2001
    '''
    if lamb < 0 or lamb > 2:
        raise ValueError('unstable relaxation parameter')
    if max_iter < 2:
        raise ValueError('unusable maximum number of iterations')
# %% user parameters
    residual = None  # init

    n = A.shape[1]  # only need rows

    if x0 is None:  # we'll use zeros
        x0 = np.zeros(n, order='F')  # 1-D vector

    if stop_mdp and taudelta == 0:
        logging.warning('tauDelta = 0 effectively disables Morozov discrepancy principle')
# %% disregard all-zero columns of A
    if issparse(A):
        A = A.tocsr()  # save time if it was csc sparse
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = np.asarray(A.multiply(A).sum(axis=1)).squeeze()  # 50 times faster than dense for 1024 x 100000 A
    else:  # is dense A
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = norm(A, ord=2, axis=1)**2  # timeit same for norm() and A**2.sum(axis=1)

    goodRows = np.unique(A.nonzero()[0])

    x = x0.copy()  # we'll leave the original x0 alone, and make a copy in x

    for i in range(max_iter):  # for each iteration
        for iRow in goodRows:  # only not all-zero rows
            # denominator AND numerator are scalar!
            # den = np.linalg.norm(A[iRow,:],2)**2
            # print(RowNormSq[iRow] == den)
            # num = ( b[iRow] - A[iRow,:] @ x )
            # x += (lamb * num/den) @ A[iRow,:]
            x += lamb * (b[iRow] - A[iRow, :] @ x) / RowNormSq[iRow] * A[iRow, :]  # first two terms are scalar always

            if nonneg:
                x[x < 0] = 0

        # handle stop rule
        if stop_mdp:
            residual = b - A @ x
            residualNorm = norm(residual, 2)
            if residualNorm <= taudelta:
                break
        if i % 200 == 0:
            residualNorm = norm(b - A @ x, 2)
            print('Iteration {},  ||residual|| = {:.2f}'.format(i, residualNorm))

    return x, residual
