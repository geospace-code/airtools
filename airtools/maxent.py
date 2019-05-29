from numpy import atleast_2d, log, atleast_1d, zeros, ones, empty, spacing, array
from numpy.linalg import norm
from warnings import warn

'''
Michael Hirsch port of P.C. Hansen Matlab ReguTools code
MAXENT Maximum entropy regularization.

 x_lambda,rho,eta = maxent(A,b,lambda,w,x0)

 Maximum entropy regularization:
    min { || A x - b ||^2 + lambda^2*x'*log(diag(w)*x) } ,
 where -x'*log(diag(w)*x) is the entropy of the solution x.
 If no weights w are specified, unit weights are used.

 If lambda is a vector, then x_lambda is a matrix such that
    x_lambda = [x_lambda(1), x_lambda(2), ... ] .

 This routine uses a nonlinear conjugate gradient algorithm with "soft"
 line search and a step-length control that insures a positive solution.
 If the starting vector x0 is not specified, then the default is
    x0 = norm(b)/norm(A,1)*ones(n,1) .

 Per Christian Hansen, IMM and Tommy Elfving, Dept. of Mathematics,
 Linkoping University, 06/10/92.

 Reference: R. Fletcher, "Practical Methods for Optimization",
 Second Edition, Wiley, Chichester, 1987.
'''


def maxent(A, b, lamb, w=None, x0=None):
    A = atleast_2d(A)

    if A.ndim != 2:
        raise ValueError('A is expected to be a 2-D Numpy array')

    b = atleast_1d(b)
    b = b.squeeze()

    if b.ndim > 1:
        raise ValueError('b is expected to be a 1-D Numpy array')

    lamb = atleast_1d(lamb)

    if lamb.ndim != 1 or lamb.size < 0:
        raise ValueError('lamb must be a 1-D vector or scalar')

    if (lamb < 0).all():
        raise ValueError('Regularization parameter lamb must be positive')


# %% Set defaults.
    flat = 1e-3     # Measures a flat minimum.
    flatrange = 10  # How many iterations before a minimum is considered flat.
    maxit = 150     # Maximum number of CG iterations.
    minstep = 1e-12  # Determines the accuracy of x_lambda.
    sigma = 0.5     # Threshold used in descent test.
    tau0 = 1e-3    # Initial threshold used in secant root finder.

# %% Initialization.
    m, n = A.shape
    lamb = atleast_1d(lamb)
    Nlambda = lamb.size

    x_lambda = zeros((n, Nlambda), order='F')
    F = zeros(maxit)

    if w is None:
        w = ones(n, dtype=float)  # needs to be column vector

    if x0 is None:
        x0 = ones(n, dtype=float)  # needs to be column vector

    rho = empty(Nlambda, dtype=float)
    eta = empty(Nlambda, dtype=float)

# Treat each lambda separately.
    for j in range(Nlambda):

        # Prepare for nonlinear CG iteration.
        l2 = lamb[j]**2.
        x = x0
        Ax = A.dot(x)
        g = 2.*A.T.dot(Ax - b) + l2*(1 + log(w*x))
        p = -g
        r = Ax - b

        # Start the nonlinear CG iteration here.
        delta_x = x
        dF = 1
        it = 0
        phi0 = p.T.dot(g)
        data = zeros((maxit, 3), dtype=float, order='F')
        X = zeros((n, maxit), dtype=float, order='F')

        while (norm(delta_x, 2) > minstep*norm(x, 2) and dF > flat and it < maxit and phi0 < 0):
            # Compute some CG quantities.
            Ap = A.dot(p)
            gamma = Ap.T.dot(Ap)
            v = A.T.dot(Ap)

            # Determine the steplength alpha by "soft" line search in which
            # the minimum of phi(alpha) = p'*g(x + alpha*p) is determined to
            # a certain "soft" tolerance.
            # First compute initial parameters for the root finder.
            alpha_left = 0
            phi_left = phi0
            if p.min() >= 0:
                alpha_right = -phi0/(2*gamma)
                h = 1. + alpha_right*p/x
            else:
                # Step-length control to insure a positive x + alpha*p.
                i = p < 0
                alpha_right = (-x[i] / p[i]).min()
                h = 1. + alpha_right*p / x
                delta = spacing(1)  # replacement for matlab eps
                while h.min() <= 0:
                    alpha_right = alpha_right*(1 - delta)
                    h = 1 + alpha_right*p / x
                    delta = delta*2

            z = log(h)
            phi_right = phi0 + 2*alpha_right*gamma + l2*p.T.dot(z)
            alpha = alpha_right
            phi = phi_right

            if phi_right <= 0:  # Special treatment of the case when phi(alpha_right) = 0.
                z = log(1 + alpha*p/x)
                g_new = g + l2*z + 2*alpha*v
                t = g_new.T.dot(g_new)
                beta = (t - g.T.dot(g_new))/(phi - phi0)
            else:
                # The regular case: improve the steplength alpha iteratively
                # until the new step is a descent step.
                t = 1
                u = 1
                tau = tau0
                uit = 0
                while u > -sigma*t:
                    uold = u
                    # Use the secant method to improve the root of phi(alpha) = 0
                    # to within an accuracy determined by tau.
                    phiit = 0
                    while abs(phi/phi0) > tau:
                        phiold = phi
                        alphaold = alpha
                        alpha = (alpha_left*phi_right - alpha_right*phi_left) / (phi_right - phi_left)
                        z = log(1 + alpha*p/x)
                        phi = phi0 + 2*alpha*gamma + l2*p.T.dot(z)
                        if phiold == phi and alphaold == alpha and phiit > maxit:
                            warn('secant is not converging: abs(phi/phi0) = ' +
                                 str(abs(phi/phi0)) +
                                 '  terminating phi search on iteration ' + str(phiit))
                            break
                        if phi > 0:
                            alpha_right = alpha
                            phi_right = phi
                        else:
                            alpha_left = alpha
                            phi_left = phi
                        phiit += 1
                    # To check the descent step, compute u = p'*g_new and
                    # t = norm(g_new)^2, where g_new is the gradient at x + alpha*p.
                    g_new = g + l2*z + 2*alpha*v
                    t = g_new.T.dot(g_new)
                    beta = (t - g.T.dot(g_new))/(phi - phi0)
                    u = -t + beta*phi
                    if u == uold and uit > maxit:
                        warn('excessive descent iterations, terminating search on iteration ' + str(phiit))
                        break
                    tau = tau/10.
                    uit += 1
            # Update the iteration vectors.
            g = g_new
            delta_x = alpha*p
            x = x + delta_x
            p = -g + beta*p
            r = r + alpha*Ap
            phi0 = p.T.dot(g)

            # Compute some norms and check for flat minimum.
            rho[j] = norm(r)
            eta[j] = x.T.dot(log(w*x))
            F[it] = rho[j]**2 + l2*eta[j]
            if it <= flatrange:
                dF = 1.
            else:
                dF = abs(F[it] - F[it-flatrange])/abs(F[it])

            data[it, ...] = array([F[it], norm(delta_x), norm(g)])
            X[..., it] = x

            it += 1

        x_lambda[..., j] = x

    return x_lambda.squeeze(), rho, eta
