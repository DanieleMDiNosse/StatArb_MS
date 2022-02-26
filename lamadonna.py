cpdef float complete_loglikelihood(cnp.ndarray params, cnp.ndarray X):
    '''Compute the total reduced log-likelihood that will be use for ML estimation'''
    cdef int T
    T = X.shape[0]
    cdef cnp.ndarray loglikelihood
    loglikelihood = np.zeros(shape=T)
    cdef cnp.ndarray b
    b = np.ones(shape=T)
    cdef float
    sigma = 1
    omega, a, alpha, beta = params[0], params[1], params[2], params[3]
    cdef int i
    for i in range(1, T - 1):
            loglikelihood[i] = - np.log(sigma) - (X[i] - a - b[i] * X[i - 1]) * (X[i] - a - b[i] * X[i - 1]) / sigma**2
            b[i + 1] = omega + alpha * (X[i] - a - b[i] * X[i - 1]) * X[i-1] / sigma**2 + beta * b[i]
    return -loglikelihood.sum()
