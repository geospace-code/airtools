import logging
import numpy as np
'''
PICARD Visual inspection of the Picard condition.

 eta = picard(U,s,b,d)
 eta = picard(U,sm,b,d)  ,  sm = [sigma,mu]

 Plots the singular values, s(i), the abs. value of the Fourier
 coefficients, |U(:,i)'*b|, and a (possibly smoothed) curve of
 the solution coefficients eta(i) = |U(:,i)'*b|/s(i).

 If s = [sigma,mu], where gamma = sigma./mu are the generalized
 singular values, then this routine plots gamma(i), |U(:,i)'*b|,
 and (smoothed) eta(i) = |U(:,i)'*b|/gamma(i).

 The smoothing is a geometric mean over 2*d+1 points, centered
 at point # i. If nargin = 3, then d = 0 (i.e, no smothing).

 Reference: P. C. Hansen, "The discrete Picard condition for
 discrete ill-posed problems", BIT 30 (1990), 658-672.

 Per Christian Hansen, IMM, April 14, 2001.
 ported to Python by Michael Hirsch
'''


def picard(U, s, b, d=0) -> tuple:

    n, ps = np.atleast_2d(s).T.shape

    beta = np.abs(np.asfortranarray(U[:, :n]).T.dot(b))
    eta = np.zeros(n, order='F')

    if ps == 2:
        s = s[:, 0] / s[:, 1]

    d21 = 2 * d + 1
    keta = np.arange(d, n-d)

    if (s == 0).any():  # 10**-14 is OK?
        logging.warning('** picard: Division by zero: singular values')

    for i in keta:
        es = np.s_[i-d:i+d+1]
        eta[i] = (beta[es].prod()**(1/d21)) / s[i]

    return eta, n, s, beta, keta, ps


def plotpicard(n, s, beta, eta, keta, ps):
    from matplotlib.pyplot import figure
    ni = np.arange(n)
    ax = figure().gca()
    ax.semilogy(ni, s, '.-')  # breaks for inf
    ax.semilogy(ni, beta, 'x')  # breaks for inf
    ax.semilogy(keta, eta[keta], 'o')
    ax.set_xlabel('i')
    ax.set_title('Picard plot')
    # ax.autoscale(True,tight=True)
    if ps == 1:
        ax.legend((r'$\sigma_i$', r'$|u_i^T b|$', r'$|u_i^T b|/\sigma_i$'), loc='lower left')
    else:
        ax.legend((r'$\sigma_i/\mu_i$', r'$|u_i^T b|$', r'$|u_i^T b|/ (\sigma_i/\mu_i)$'), loc='lower left')
