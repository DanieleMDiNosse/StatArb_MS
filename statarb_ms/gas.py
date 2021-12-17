import numpy as np
from scipy.optimize import minimize
from makedir import go_up
import matplotlib.pyplot as plt
import time
import argparse
from numba import njit, prange
from tqdm import tqdm
import math


@njit(parallel=True)
def reduced_loglikelihood(params, X, sigma):
    '''Compute the total reduced log-likelihood that will be use for ML estimation'''
    T = X.shape[0]
    loglikelihood = np.zeros(shape=T)
    b = np.ones(shape=T)
    xi = np.zeros(shape=T)
    if params.shape[0] == 5:
        alpha, beta, omega, sigma, a = params[0], params[1], params[2], params[3], params[4]
    else:
        alpha, beta, omega, a = params[0], params[1], params[2], params[3]

    for i in prange(1, T - 1):
        loglikelihood[i] = -0.5 * ((T - 2) * np.log(sigma**2) +
                                   1 / (sigma**2) * (X[i] - b[i] * X[i - 1])**2)
        b[i + 1] = omega + alpha * (X[i - 1] * xi[i] / sigma**2) + beta * b[i]
        xi[i + 1] = X[i + 1] - a - b[i + 1] * X[i]

    return -loglikelihood.sum() / T


def estimation(fun, X, n_params, method='BFGS', verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.ones(shape=T)
    xi = np.zeros(shape=T)
    init_params = np.random.uniform(0, 1, size=n_params)
    if n_params == 4: sigma = 1
    res = minimize(fun, init_params, (X, sigma),
                   method=method, jac='3-point', options={'maxiter': 800})
    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(f'Estimated parameters: \n Message: {res.message} \n Success: {res.success} \n Estimate: {res.x} \n')
    if n_params == 5:
        alpha, beta, omega, sigma, a = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]
    else:
        alpha, beta, omega, a = res.x[0], res.x[1], res.x[2], res.x[3]

    for i in range(1, T - 1):
        b[i + 1] = omega + alpha * (X[i - 1] * xi[i] / sigma**2) + beta * b[i]
        xi[i + 1] = X[i + 1] - a - b[i + 1] * X[i]

    if visualization:
        X_est = np.zeros_like(X)
        for i in range(X.shape[0] - 1):
            X_est[i+1] = a + b[i]*X_est[i] + xi[i+1]
        plt.figure()
        plt.plot(X, label='True')
        plt.plot(X_est, label='Estimated')
        plt.legend()
        plt.show()

    return b, a, xi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-m", "--method", type=int,
                        help="0 for BFGS, 1 for Nelder-Mead.")
    parser.add_argument("-n", "--n_params", type=int,
                        help="Number of parameters to be estimated.")
    args = parser.parse_args()
    plt.style.use('seaborn')
    start = time.time()
    np.random.seed(666)

    if args.method == 0:
        method = 'BFGS'
    if args.method == 1:
        method = 'Nelder-Mead'

    X = np.load(go_up(1) + '/saved_data/dis_res.npy')
    days = X.shape[0]
    n_stocks = X.shape[1]
    time_list = []
    for day in range(100):
        for stock in range(n_stocks):
            start = time.time()
            x = X[day, stock, :]
            b, a, xi = estimation(
                reduced_loglikelihood, x, n_params=args.n_params, method=method, verbose=True, visualization=True)
            if (math.isnan(b[-1])) or (b[-1] < 0):
                print(b[-1])
            end = time.time()
            time_list.append(end-start)

    print(np.array(time_list).mean())
    end = time.time()
    time_elapsed = (end - start)
    print('Elapsed time: %.2f seconds' % time_elapsed)


#     fig = plt.figure()
#     ax = plt.gca(projection='3d')
#     ax.plot_surface(x, y, plot_loglikelihood)
#     plt.show()
