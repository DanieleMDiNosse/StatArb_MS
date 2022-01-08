import numpy as np
from scipy.optimize import minimize, least_squares
from makedir import go_up
import matplotlib.pyplot as plt
import time
import argparse
from numba import njit, prange
from tqdm import tqdm
import math
import reduced_loglikelihood


def estimation(fun, X, method, update, verbose=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.ones(shape=T)
    xi = np.zeros(shape=T)
    init_params = np.random.uniform(0, 1, size=4)
    sigma = 1
    sq_sgm = sigma * sigma
    # res = least_squares(fun, init_params, args=(X, sigma, update))
    res = minimize(fun, init_params, (X, sigma, update),
                   method=method, options={'xatol': 0.001, 'fatol': 0.001})
    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(1.5)
    omega, a, alpha, beta = res.x[0], res.x[1], res.x[2], res.x[3]

    for i in range(1, T - 1):
        if update == 'gaussian':
            b[i + 1] = omega + alpha * xi[i] * X[i-1] / sq_sgm + beta * b[i]
            xi[i + 1] = X[i + 1] - a - b[i + 1] * X[i]
        if update == 'logistic':
            b[i + 1] = 0.5 * (omega + alpha * ((X[i] - 1 / (1 + np.exp(-b[i])) * X[i - 1]) * np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * X[i - 1] / sq_sgm) + beta * b[i])
            xi[i + 1] = X[i + 1] - a - 1 / (1 + np.exp(-b[i + 1])) * X[i]
        if update == 'logarithm':
            b[i + 1] = omega + alpha * X[i - 1] / (b[i] * sq_sgm) * (X[i] - a - np.log(b[i]) * X[i-1]) + beta * b[i]
            xi[i + 1] = X[i + 1] - a - np.log(b[i + 1]) * X[i]

    if update == 'logistic': b = 1 / (1 + np.exp(-b))
    if update == 'logarithm': b = np.log(b)

    # if visualization:
    #     X_est = np.zeros_like(X)
    #     for i in range(X.shape[0] - 1):
    #         X_est[i + 1] = a + b[i + 1]*X_est[i] + xi[i + 1]
    #     plt.figure()
    #     plt.plot(X, label='True')
    #     plt.plot(X_est, label='Estimated')
    #     plt.legend()
    #     plt.show()

    return b, a, xi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-m", "--method", type=int,
                        help="0 for BFGS, 1 for Nelder-Mead.")
    parser.add_argument("-u", "--update", type=int, help="Update scheme for gas estimation")
    parser.add_argument("-v", "--verbose", action='store_true')

    args = parser.parse_args()
    plt.style.use('seaborn')
    np.random.seed(666)

    if args.method == 0:
        method = 'BFGS'
    if args.method == 1:
        method = 'Nelder-Mead'

    if args.update == 0:
        update = 'gaussian'
    if args.update == 1:
        update = 'logistic'
    if args.update == 2:
        update = 'logarithm'

    X = np.load(go_up(1) + '/saved_data/dis_res.npy')
    days = X.shape[0]
    n_stocks = X.shape[1]
    time_list = []
    for day in range(100):
        start = time.time()
        for stock in range(n_stocks):
            x = X[day, stock, :]
            b, a, xi = estimation(
                reduced_loglikelihood.reduced_loglikelihood, x, update=update, method=method, verbose=args.verbose)
            if (math.isnan(b[-1])) or (b[-1] < 0):
                print(b[-1])
        end = time.time()
        print(f'time day {day}: ', end-start)


#     fig = plt.figure()
#     ax = plt.gca(projection='3d')
#     ax.plot_surface(x, y, plot_loglikelihood)
#     plt.show()
