cimport numpy as cnp
import numpy as np
cimport cython
import time

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float reduced_loglikelihood(cnp.ndarray params, cnp.ndarray X, float sigma, str update):
    '''Compute the total reduced log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray loglikelihood
    loglikelihood = np.zeros(shape=T)
    cdef cnp.ndarray b
    b = np.ones(shape=T)
    cdef cnp.ndarray xi
    xi = np.zeros(shape=T)
    cdef float sq_sgm
    sq_sgm = sigma * sigma
    omega, a, alpha, beta = params[0], params[1], params[2], params[3]

    cdef int i
    for i in range(1, T - 1):
        if update == 'gaussian':
            loglikelihood[i] = (X[i] - a - b[i] * X[i - 1])**2
            b[i + 1] = omega + alpha * xi[i] * X[i-1] / sq_sgm + beta * b[i]
        if update == 'logistic':
            loglikelihood[i] = (X[i] - a - 1 / (1 + np.exp(-b[i])) * X[i - 1])**2
            s = (X[i] - 1 / (1 + np.exp(-b[i])) * X[i - 1]) * np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * X[i - 1] / sq_sgm
            b[i + 1] = omega + alpha * s + beta * b[i]
        if update == 'logarithm':
            loglikelihood[i] = (X[i] - a - np.log(b[i]) * X[i - 1])**2
            b[i + 1] = omega + alpha * X[i - 1] / (b[i] * sq_sgm) * (X[i] - a - np.log(b[i]) * X[i-1]) + beta * b[i]

    return 0.5 * (1 / sq_sgm) * loglikelihood.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float SSR(cnp.ndarray params, cnp.ndarray X, float sigma):
    '''Compute the total reduced log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray loglikelihood
    loglikelihood = np.zeros(shape=T)
    cdef cnp.ndarray b
    b = np.ones(shape=T)
    cdef cnp.ndarray xi
    xi = np.zeros(shape=T)
    alpha, beta, omega, a = params[0], params[1], params[2], params[3]

    cdef int i
    for i in range(1, T - 1):
        loglikelihood[i] = (X[i] - X[i - 1] / (1 + np.exp(-b[i])))**2
        b[i + 1] = omega + alpha * (X[i - 1] * np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * xi[i] / sigma**2) + beta * b[i]
        xi[i + 1] = X[i + 1] - a - 1 / (1 + np.exp(-b[i + 1])) * X[i]

    return -loglikelihood.sum() / T
