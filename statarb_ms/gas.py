import numpy as np
from scipy.optimize import minimize
from makedir import go_up
import matplotlib.pyplot as plt
import time
import argparse
from numba import njit, prange
from tqdm import tqdm
import math
import reduced_loglikelihood
from tqdm import tqdm


def estimation(fun, X, init_params, method='Nelder-Mead', update='gaussian', verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.ones(shape=T)
    xi = np.zeros(shape=T)
    X_est = np.zeros_like(X)
    sigma = 1
    n_params = init_params.shape[0]
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
            b[i + 1] = omega + alpha * xi[i] * \
                X[i - 1] / sigma**2 + beta * b[i]
            xi[i + 1] = X[i + 1] - a - b[i + 1] * \
                X[i]  # prediction error decomposition
        if update == 'logistic':
            b[i + 1] = omega + alpha * ((X[i] - 1 / (1 + np.exp(-b[i])) * X[i - 1]) *
                                        np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * X[i - 1] / sq_sgm) + beta * b[i]
            # prediction error decomposition
            xi[i + 1] = X[i + 1] - a - 1 / (1 + np.exp(-b[i + 1])) * X[i]

    if update == 'logistic':
        b = 1 / (1 + np.exp(-b))

    if visualization:
        X_est = np.zeros_like(X)
        b_est = np.zeros_like(b)
        for t in range(1, X_est.shape[0] - 1):
            X_est[t + 1] = a + b_est[t + 1] * X_est[t] + xi[t + 1]
            b_est[t + 1] = omega + alpha * xi[t] * \
                X_est[t - 1] + beta * b_est[t]
        plt.figure(figsize=(12, 8))
        plt.plot(X, label='Data')
        plt.plot(X_est, label='Filtered Data')
        plt.legend()
        plt.grid(True)
        plt.show()

    return b, a, xi, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-u", "--update", type=int, default=0,
                        help="Update scheme for gas estimation")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-vv", "--visualization", action='store_true')
    parser.add_argument("-c", "--convergence", action='store_true')

    args = parser.parse_args()
    plt.style.use('seaborn')
    # np.random.seed(666)

    if args.update == 0:
        update = 'gaussian'
    if args.update == 1:
        update = 'logistic'

    if args.convergence:
        X = np.load(go_up(1) + '/saved_data/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        for day in range(100):
            start = time.time()
            for stock in range(n_stocks):
                x = X[day, stock, :]
                fun_val = []
                b_val = []
                est_par = np.empty(shape=(100, args.n_params))
                par = np.random.uniform(0, 1, size=(100, args.n_params))
                for i in range(len(par)):
                    b, a, xi, res = estimation(
                        reduced_loglikelihood.reduced_loglikelihood, x, par[i], update=update, verbose=args.verbose, visualization=args.visualization)
                    fun_val.append(res.fun)
                    est_par[i] = res.x
                    b_val.append(b[-1])
                    # add the difference between the initial guess and the estimated parameters
                plt.figure(figsize=(12, 8), tight_layout=True)
                ax0 = plt.subplot(2, 3, 4)
                ax0.plot(b_val, 'k')
                ax0.title.set_text('b(60) values')
                ax1 = plt.subplot(2, 3, 1)
                ax1.plot(fun_val, 'k')
                ax1.title.set_text('Likelihood Evaluations')
                ax2 = plt.subplot(args.n_params, 3, 2)
                ax2.plot(par[:, 0])
                ax2.title.set_text('Initial Omega')
                ax3 = plt.subplot(args.n_params, 3, 5)
                ax3.plot(par[:, 1])
                ax3.title.set_text('Initial a')
                ax4 = plt.subplot(args.n_params, 3, 8)
                ax4.plot(par[:, 2])
                ax4.title.set_text('Initial alpha')
                ax5 = plt.subplot(args.n_params, 3, 11)
                ax5.plot(par[:, 3])
                ax5.title.set_text('Initial beta')

                ax6 = plt.subplot(args.n_params, 3, 3)
                ax6.plot(est_par[:, 0])
                ax6.title.set_text('Estimated Omega')
                ax7 = plt.subplot(args.n_params, 3, 6)
                ax7.plot(est_par[:, 1])
                ax7.title.set_text('Estimated a')
                ax8 = plt.subplot(args.n_params, 3, 9)
                ax8.plot(est_par[:, 2])
                ax8.title.set_text('Estimated alpha')
                ax9 = plt.subplot(args.n_params, 3, 12)
                ax9.plot(est_par[:, 3])
                ax9.title.set_text('Estimated beta')
                plt.grid(True)
                plt.show()
                if (math.isnan(b[-1])) or (b[-1] < 0):
                    print(b[-1])
            end = time.time()
            print(f'time day {day}: ', end - start)

    else:
        X = np.load(go_up(1) + '/saved_data/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        for day in range(100):
            start = time.time()
            for stock in range(n_stocks):
                x = X[day, stock, :]
                init_params = np.random.uniform(0, 1, size=args.n_params)
                b, a, xi, fun = estimation(
                    reduced_loglikelihood.reduced_loglikelihood, x, n_params=args.n_params, update=update, verbose=args.verbose, visualization=args.visualization)
                if (math.isnan(b[-1])) or (b[-1] < 0):
                    print(b[-1])
            end = time.time()
            print(f'time day {day}: ', end - start)

#     fig = plt.figure()
#     ax = plt.gca(projection='3d')
#     ax.plot_surface(x, y, plot_loglikelihood)
#     plt.show()
