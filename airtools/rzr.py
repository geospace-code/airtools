
def rzr(A, b=None, Nthr=0):
    '''
    rzr  Remove zero rows of A and the corresponding elements of b.

    Assumes A,b are Numpy arrays

     A,b = rzr(A,b)
     A,b = rzr(A,b,Nthr)

     Identifies zero rows of the coefficient matrix A and removes them.
     If a right-hand side b is present, the corresponding elements of
     b are also removed.

     If a positive Nthr is given as the third argument, then all rows with
     less than or equal to Nthr nonzero elements are removed.

     Use this function to 'clean up' a discretized tomography problem.
     Zero rows do not contribute to the reconstruction.
     Rows with few nonzero elements correspond to pixels near the corners of
     the image, whose reconstructions are highly sensitive to noise.

     Per Chr. Hansen, October 11, 2011, DTU Compute.
     ported to Python by Michael Hirsch
    '''

    s = (A > 0).sum(axis=1)  # number of non-zero elements per row
    goodInd = s > Nthr
    # ngood = goodInd.sum()
    A = A[goodInd, :]

    if b is not None:
        b = b[goodInd]

    return A, b, goodInd
