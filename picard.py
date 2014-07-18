import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
#PICARD Visual inspection of the Picard condition.
#
# eta = picard(U,s,b,d)
# eta = picard(U,sm,b,d)  ,  sm = [sigma,mu]
#
# Plots the singular values, s(i), the abs. value of the Fourier
# coefficients, |U(:,i)'*b|, and a (possibly smoothed) curve of
# the solution coefficients eta(i) = |U(:,i)'*b|/s(i).
#
# If s = [sigma,mu], where gamma = sigma./mu are the generalized
# singular values, then this routine plots gamma(i), |U(:,i)'*b|,
# and (smoothed) eta(i) = |U(:,i)'*b|/gamma(i).
#
# The smoothing is a geometric mean over 2*d+1 points, centered
# at point # i. If nargin = 3, then d = 0 (i.e, no smothing).

# Reference: P. C. Hansen, "The discrete Picard condition for
# discrete ill-posed problems", BIT 30 (1990), 658-672.

# Per Christian Hansen, IMM, April 14, 2001.
# ported to Python by Michael Hirsch
def picard(U,s,b,d=0):
    n,ps = np.atleast_2d(s).T.shape # the transpose is because Numpy 1.8.1 has no order='F' option for atleast2d

    beta = np.abs( U[:,:n].T.dot(b) )
    eta = np.zeros(n,order='F')

    if ps==2: s = s[:,0] / s[:,1]

    d21 = 2 * d + 1
    keta = np.arange(d,n-d)

    if np.any(np.isclose(s,0)):
        warn('picard: Division by zero: singular values')

    #for ik,k in enumerate(keta):
    #    eta[ik] = ( np.prod(beta[ik-d:ik+d])**(1/d21)) /s[ik]
    for i in keta:
        es = np.s_[i-d:i+d+1]
        eta[i] = ( np.prod(beta[es])**(1/d21)) / s[i]
#%% plot Picard plot
    ni = np.arange(n)
    plt.figure()
    plt.semilogy(ni, s, '.-', ni, beta, 'x', keta, eta[keta], 'o')
    plt.xlabel('i')
    plt.title('Picard plot')
    if ps==1:
        plt.legend( ('$\sigma_i$','$|u_i^T b|$','$|u_i^T b|/\sigma_i$'),loc='lower left' )
    else:
        plt.legend( ('$\sigma_i/\mu_i$','$|u_i^T b|$','$|u_i^T b|/ (\sigma_i/\mu_i)$') ,loc='lower left')


    return eta
