#!/usr/bin/env python3
from __future__ import division
from numpy import zeros, where,unique,asarray
from numpy.linalg import norm
from scipy.sparse import issparse

def kaczmarz_ART(A,b,maxIter=8,x0=None,lambdaRelax=1,stopmode=None,taudelta=0,nonneg=True,dbglvl=0):
    # TODO: add randomized ART, and other variants
    # Michael Hirsch May 2014
    # GPL v3+ license
    #
    # tested with Dense and Sparse arrays. (Aug 2014)
    #
    # inputs:
    # A:  M x N 2-D projection matrix
    # b:  N x 1 1-D vector of observations
    # maxIter: maximum number of ART iterations
    # x0: N x 1 1-D vector of initialization (a guess at x)
    # lambdaRelax: relaxation parameter (see Herman Ch.11.2)
    # stopmode: {None, MDP}  stop before maxIter if solution is good enough
    #             (MDP is Morozov Discrepancy Principle)
    # nonneg: enforces non-negativity of solution
    #
    # outputs:
    # x: the estimated solution of A x = b
    # residual: the error b-Ax
    #
    # References:
    # Herman, G. " Fundamentals of Computerized Tomography", 2nd Ed., Springer, 2009
    # Natterer, F. "The mathematics of computed tomography", SIAM, 2001

#%% user parameters
    residual = None #init

    if dbglvl>0:
        print(('Lambda Relaxation: ' + str(lambdaRelax)))

    n = A.shape[1] #only need rows

    if x0 is None: # we'll use zeros
        print('kaczmarz: using zeros to initialize x0')
        x0 = zeros(n,order='F') #1-D vector

    if stopmode is None or stopmode.lower() =='iter': # just use number of iterations
        sr = 0
    elif stopmode.lower() == 'mdp' or stopmode.lower()== 'dp':
        sr = 1
        if taudelta==0: print('you used tauDelta=0, which effectively disables Morozov discrepancy principle')
    else:
        sr = 0
        print("didn't understand stopmode command, defaulted to maximum iterations")

#%% disregard all-zero columns of A
    if issparse(A):
        A = A.tocsr() #save time if it was csc sparse
        goodRows = unique(A.nonzero()[0])
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = asarray(A.multiply(A).sum(axis=1)).squeeze() # 50 times faster than dense for 1024 x 100000 A
    else: #is dense A
        goodRows = where( (A>0).any(axis=1) )[0] #we want indices
        # speedup: compute norms along columns at once, and retrieve
        RowNormSq = norm(A,ord=2,axis=1)**2 #timeit same for norm() and A**2.sum(axis=1)

    x = x0.copy() # we'll leave the original x0 alone, and make a copy in x
    iIter = 0
    stop = False #FIXME will always run at least once
    while not stop: #for each iteration
        for iRow in goodRows:  #only not all-zero rows
            #denominator AND numerator are scalar!
            #den = np.linalg.norm(A[iRow,:],2)**2
            #print(RowNormSq[iRow] == den)
            #num = ( b[iRow] - A[iRow,:].dot(x) )
            #x = x + np.dot( lambdaRelax * num/den , A[iRow,:] )
            x += lambdaRelax * ( b[iRow] - A[iRow,:].dot(x) ) / RowNormSq[iRow] *  A[iRow,:] #first two terms are scalar always

            if nonneg: x[x<0] = 0


        iIter += 1
        #handle stop rule
        stop = iIter > maxIter
        if sr == 0: # no stopping till iterations are done
            pass
        elif sr == 1:
            residual = b - A.dot(x)
            residualNorm = norm(residual,2)
            stop |= (residualNorm <= taudelta)
        if iIter % 200 == 0: #print update every N loop iterations for user comfort
            print( ('kaczmarz: Iteration ' + str(iIter) + ',  ||residual|| = ' + str(residualNorm) ) )
    return x,residual,iIter-1

if __name__ == '__main__':
    from numpy.testing import assert_array_almost_equal
    print('selftest mode')
    A = asarray([[1, 2, 0],[0, 4, 3]])
    b = asarray([8,18])
    x = kaczmarz_ART(A,b,50)[0]
    assert_array_almost_equal(x,[ 0.91803279,  3.54098361,  1.27868852])
    exit(0)
