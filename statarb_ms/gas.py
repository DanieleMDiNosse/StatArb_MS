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

def model():
    s = np.zeros(shape=500)
    sgm = np.array([2 + 0.5 * np.sin(np.pi * t / 100) for t in range(s.shape[0])])
    for t in range(s.shape[0]):
        s[t] = sgm[t] * np.random.normal()
    return s, sgm

def model_loglikelihood(params, s):
    loglikelihood = np.zeros_like(s)
    var = np.zeros_like(s)
    var[0] = 0.1
    T = s.shape[0]
    omega, alpha, beta = params[0], params[1], params[2]
    for i in range(T-1):
        loglikelihood[i] = -0.5 * (np.log(2*np.pi*var[i]) + s[i]**2/var[i])
        var[i + 1] = np.abs(omega) + np.abs(alpha) * (s[i]**2 - var[i]) + np.abs(beta) * var[i]
    return -loglikelihood.sum()

def model_estimation(fun, s, sgm):
    init_params = np.random.uniform(0, 0.1, size=3)
    print(f'Initial guess: \n {init_params}')
    res = minimize(fun, init_params, s, method='BFGS', options={'gtol': 1e-5})
    omega, alpha, beta = res.x[0], res.x[1], res.x[2]
    print(res)
    var = np.ones_like(s)
    s = np.ones_like(s)
    for i in range(1, s.shape[0]):
        var[i] = omega + alpha * (s[i-1]**2 - var[i-1]) + beta * var[i-1]
        s[i] = np.sqrt(var[i]) * np.random.normal()
    plt.figure()
    plt.scatter(np.linspace(0,s.shape[0], s.shape[0]), s, s=4, label='Filtered values')
    plt.plot(sgm, label='Real dynamics')
    plt.grid(True)
    plt.tight_layout(True)
    plt.show()


def estimation(fun, X, n_params, method='Nelder-Mead', update='gaussian', verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.ones(shape=T)
    xi = np.zeros(shape=T)
    X_est = np.zeros_like(X)
    sigma = 1
    if n_params == 4:
        init_params = np.random.uniform(0, 1, size=4)
    if n_params == 5:
        init_params = np.random.uniform(0,1, size=5)
    res = minimize(fun, init_params, (X, n_params, update, sigma),
                   method=method)
    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(1.5)
    if n_params == 4:
        omega, a, alpha, beta = res.x[0], res.x[1], res.x[2], res.x[3]
    if n_params == 5:
        omega, a, alpha, beta, sigma = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]

    for i in range(1, T - 1):
        if update == 'gaussian':
            b[i + 1] = omega + alpha * xi[i] * X[i-1] / sigma**2 + beta * b[i]
            xi[i + 1] = X[i + 1] - a - b[i + 1] * X[i] # prediction error decomposition
        if update == 'logistic':
            b[i + 1] = omega + alpha * ((X[i] - 1 / (1 + np.exp(-b[i])) * X[i - 1]) * np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * X[i - 1] / sq_sgm) + beta * b[i]
            xi[i + 1] = X[i + 1] - a - 1 / (1 + np.exp(-b[i + 1])) * X[i] # prediction error decomposition
        if update == 'logarithm':
            b[i + 1] = omega + alpha * X[i - 1] / (b[i] * sq_sgm) * (X[i] - a - np.log(b[i]) * X[i-1]) + beta * b[i]
            xi[i + 1] = X[i + 1] - a - np.log(b[i + 1]) * X[i] # prediction error decomposition

    if update == 'logistic': b = 1 / (1 + np.exp(-b))
    if update == 'logarithm': b = np.log(b)

    if visualization:
        X_est = np.zeros_like(X)
        b_est = np.zeros_like(b)
        for t in range (1, X_est.shape[0] - 1):
            X_est[t + 1] = a + b_est[t + 1] * X_est[t] + xi[t + 1]
            b_est[t + 1] = omega + alpha * xi[t] * X_est[t - 1] + beta * b_est[t]
        plt.figure(figsize=(12,8))
        plt.plot(X, label='Data')
        plt.plot(X_est, label='Filtered Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    return b, a, xi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-u", "--update", type=int, default=0, help="Update scheme for gas estimation")
    parser.add_argument("-n", "--n_params", type=int, default=4, help="Number of parameters to be estimates")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-vv", "--visualization", action='store_true')
    parser.add_argument("-m", "--model", action='store_true')

    args = parser.parse_args()
    plt.style.use('seaborn')
    np.random.seed(666)

    if args.update == 0:
        update = 'gaussian'
    if args.update == 1:
        update = 'logistic'
    if args.update == 2:
        update = 'logarithm'

    if args.model:
        s, sgm = model()
        model_estimation(model_loglikelihood, s, sgm)

    else:
        X = np.load(go_up(1) + '/saved_data/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        for day in range(100):
            start = time.time()
            for stock in range(n_stocks):
                x = X[day, stock, :]
                b, a, xi = estimation(
                    reduced_loglikelihood.reduced_loglikelihood, x, n_params=args.n_params, update=update, verbose=args.verbose, visualization=args.visualization)
                if (math.isnan(b[-1])) or (b[-1] < 0):
                    print(b[-1])
            end = time.time()
            print(f'time day {day}: ', end-start)


#     fig = plt.figure()
#     ax = plt.gca(projection='3d')
#     ax.plot_surface(x, y, plot_loglikelihood)
#     plt.show()
