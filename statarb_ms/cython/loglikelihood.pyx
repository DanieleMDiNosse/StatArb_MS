cimport numpy as cnp
import numpy as np
cimport cython
import time
from scipy.optimize import minimize
import warnings

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float loglikelihood_after_targ(cnp.ndarray params, cnp.ndarray X, float b_bar):
    '''Compute the total log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray b
    b = np.zeros(shape=T)
    b[:2] = b_bar
    a, alpha, beta, = params
    cdef float omega
    omega = beta * (1 - b_bar)
    cdef float sigma
    sgm = 1

    cdef int i
    for i in range(1, T - 1):
        b[i + 1] = omega + alpha * (X[i] - a - b[i] * X[i - 1]) * X[i-1] / sgm**2 + beta * b[i]

    cdef float sum
    sum = 0
    for i in range(T - 1):
        sum += (- 0.5 * np.log(sigma**2) - 0.5 *
                      (X[i + 1] - a - b[i + 1] * X[i])**2 / sgm**2)
    return - sum / T

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float targeting_loglikelihood(cnp.ndarray params, cnp.ndarray X):
    '''Compute the total log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray b
    b = np.zeros(shape=T)
    omega, a = params
    cdef float sigma
    sigma = 1

    b[:] = omega

    cdef float sum
    sum = 0
    for i in range(T - 1):
        sum += (- 0.5 * np.log(sigma**2) - 0.5 *
                      (X[i + 1] - a - b[i + 1] * X[i])**2 / sigma**2)
    return - sum / T

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float loglikelihood(cnp.ndarray params, cnp.ndarray X, str link_fun):
    '''Compute the total log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray b
    b = np.zeros(shape=T)
    cdef float omega
    cdef float a
    cdef float alpha
    cdef float beta
    cdef int lam
    cdef float sum

    omega, a, alpha, beta = params
    lam = 1

    cdef int t
    for t in range(1, T - 1):
        if link_fun == 'identity':
            b[t + 1] = omega + alpha * (X[t] - a - b[t] * X[t - 1]) * X[t - 1] + beta * b[t]

        if link_fun == 'logistic':
            b[t + 1] = omega + alpha * (X[t] - a - 1 / (1 + np.exp(-b[t])) * \
                    X[t - 1]) * np.exp(-b[t]) / (1 + np.exp(-b[t]))**2 * X[t - 1] + beta * b[t]

        if link_fun == 'identity_student':
            b[t + 1] = omega + alpha * (lam + 1) * X[t - 1] * (X[t] - a - b[t] * X[t - 1])  / (lam + (X[t] - a - b[t] * X[t - 1])**2) + beta * b[t]

    sum = 0
    for t in range(T - 1):
        if link_fun == 'identity':
            sum += - 0.5 * (X[t + 1] - a - b[t + 1] * X[t]) * (X[t + 1] - a - b[t + 1] * X[t])

        if link_fun == 'logistic':
            sum += - 0.5 * (X[t + 1] - a - 1 / (1 + np.exp(-b[t + 1])) * X[t])**2

        if link_fun == 'identity_student':
            sum += - (lam + 1) * 0.5 * np.log(1 + (X[t + 1] - a - b[t + 1] * X[t])**2 / lam)

    return - sum / T
