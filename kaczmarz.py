import numpy as np
#import matplotlib.pyplot as plt

def kaczmarz_ART(A,b,maxIter=8,x0=None,lambdaRelax=1,stopmode=None,taudelta=0,nonneg=True,dbglvl=0):
    # TODO: add randomized ART, and other variants
    # Michael Hirsch May 2014
    # GPL v3+ license
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

    if dbglvl>0:
        print(('Lambda Relaxation: ' + str(lambdaRelax)))

    m,n = A.shape

    if x0 is None: # we'll use zeros
        print('kaczmarz: using zeros to initialize x0')
        x0 = np.zeros(n) #1-D vector

    if stopmode is None: # just use number of iterations
        sr = 0
    elif stopmode == 'MDP':
        sr = 1
        if taudelta==0: print('you used tauDelta=0, which effectively disables Morozov discrepancy principle')
    else:
        sr = 0
        print("didn't understand stopmode command, defaulted to maximum iterations")

#%% disregard all-zero columns of A
    goodRows = np.where( np.any(A>0,axis=1) )[0] #we want indices
#%% speedup: compute norms along columns at once, and retrieve
    RowNormSq = np.linalg.norm(A,ord=2,axis=1)**2

    x = np.copy(x0) # we'll leave the original x0 alone, and make a copy in x
    iIter = 0
    stop = False #FIXME will always run at least once
    while not stop: #for each iteration
        for iRow in goodRows:  #only not all-zero rows
        #for iRow in range(m):    #for each row
            #denominator AND numerator are scalar!
            #den = np.linalg.norm(A[iRow,:],2)**2
            #print(RowNormSq[iRow] == den)
            num = ( b[iRow] - A[iRow,:].dot(x) )
            #x = x + np.dot( lambdaRelax * num/den , A[iRow,:] )
            x = x + np.dot( lambdaRelax * num/RowNormSq[iRow] , A[iRow,:] )

            if nonneg: x[x<0] = 0

        residual = b - A.dot(x)
#            if True:
#                plt.plot(x)
#                plt.show(True)
        iIter += 1
        #handle stop rule
        stop = iIter > maxIter
        if sr == 0: # no stopping till iterations are done
            pass
        elif sr == 1:
            stop |= np.linalg.norm(residual,2) <= taudelta
    return x,residual,iIter-1
