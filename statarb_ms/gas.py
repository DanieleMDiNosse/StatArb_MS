import numpy as np
from scipy.optimize import minimize
from scipy import signal
from makedir import go_up
import matplotlib.pyplot as plt
import time
import argparse
from numba import njit, prange
from tqdm import tqdm
import math
import reduced_loglikelihood
from tqdm import tqdm

# def targeting_estimation(fun, X, init_params, method='BFGS'):
#     '''Estimation of GAS parameters'''
#     T = X.shape[0]
#     b = np.zeros(shape=T)
#     res = minimize(fun, init_params, X, method=method)
#     omega, a, sigma = res.x[0], res.x[1], res.x[2]
#
#     for i in range(1, T - 1):
#         b[i + 1] = omega
#
#     return b[-1]

def estimation(fun, X, init_params, method='Nelder-Mead', targeting_estimation=False, verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.zeros(shape=T)
    xi = np.zeros(shape=T)

    if targeting_estimation:
        init_params0 = np.random.uniform(0, 1, size=3)
        res = minimize(reduced_loglikelihood.targeting_loglikelihood, init_params0, X, method=method)
        omega, a, sigma = res.x[0], res.x[1], res.x[2]
        b_bar = omega
        b[:2] = b_bar
        res = minimize(fun, init_params, (X, b_bar), method=method)
        a, alpha, beta, sigma = res.x[0], res.x[1], res.x[2], res.x[3]
        omega = beta * (1 - b_bar)

    else:
        res = minimize(fun, init_params, X, method=method)
        omega, a, alpha, beta = res.x[0], res.x[1], res.x[2], res.x[3]
        sigma = 1

    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(2.5)

    for i in range(1, T - 1):
        b[i + 1] = omega + alpha * (X[i] - a - b[i] * X[i - 1]) * X[i - 1] / sigma**2 + beta * b[i]

    if visualization:
        # plt.plot(b)
        # plt.show()
        # X_est = np.zeros_like(X)
        # b_est = np.zeros_like(b)
        # for t in range(1, T - 1):
        #     b_est[t + 1] = omega + alpha * (X_est[i + 1] - a - b_est[i + 1] * X_est[i]) * X_est[t - 1] + beta * b_est[t]
        #     X_est[t + 1] = a + b_est[t + 1] * X_est[t] + np.random.normal(0,sigma)
        # plt.figure(figsize=(12, 8))
        # plt.plot(X, label='Data')
        plt.plot(b, label='Filtered Data')
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
    parser.add_argument("-t", "--targ_est", action='store_true')

    args = parser.parse_args()
    plt.style.use('seaborn')
    np.random.seed(666)

    if args.update == 0: update = 'gaussian'
    if args.update == 1: update = 'logistic'

    if args.convergence:
        n_params = 4
        N = 100
        X = np.load(go_up(1) + '/saved_data/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        r = np.random.randint(0,days)
        rr = np.random.randint(0, n_stocks)
        print(r, rr)
        x = X[1000, 9, :]
        fun_val = []
        b_val = []
        est_par = np.empty(shape=(N, 4))
        par = np.random.uniform(0, 1, size=(N, 4))
        for i in tqdm(range(len(par)), desc='Convergence plots'):
            b, a, xi, res = estimation(
                reduced_loglikelihood.complete_loglikelihood, x, par[i], verbose=args.verbose, visualization=args.visualization)
            fun_val.append(res.fun)
            est_par[i] = res.x
            b_val.append(b[-1])
        plt.figure(figsize=(12, 8), tight_layout=True)
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'crimson', linewidth=1)
        ax0.title.set_text('b(60) values')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'crimson', linewidth=1)
        ax1.title.set_text('Likelihood Evaluations')
        ax2 = plt.subplot(n_params, 3, 2)
        ax2.plot(par[:, 0], 'slateblue', linewidth=1)
        ax2.title.set_text('Initial Omega')
        ax2.tick_params(labelbottom=False)
        ax3 = plt.subplot(n_params, 3, 5)
        ax3.plot(par[:, 1], 'slateblue', linewidth=1)
        ax3.title.set_text('Initial a')
        ax3.tick_params(labelbottom=False)
        ax4 = plt.subplot(n_params, 3, 8)
        ax4.plot(par[:, 2], 'slateblue', linewidth=1)
        ax4.title.set_text('Initial alpha')
        ax4.tick_params(labelbottom=False)
        ax5 = plt.subplot(n_params, 3, 11)
        ax5.plot(par[:, 3], 'slateblue', linewidth=1)
        ax5.title.set_text('Initial beta')

        ax6 = plt.subplot(n_params, 3, 3)
        ax6.plot(est_par[:, 0], 'g', linewidth=1)
        ax6.title.set_text('Estimated Omega')
        ax6.tick_params(labelbottom=False)
        ax7 = plt.subplot(n_params, 3, 6)
        ax7.plot(est_par[:, 1], 'g', linewidth=1)
        ax7.title.set_text('Estimated a')
        ax7.tick_params(labelbottom=False)
        ax8 = plt.subplot(n_params, 3, 9)
        ax8.plot(est_par[:, 2], 'g', linewidth=1)
        ax8.title.set_text('Estimated alpha')
        ax8.tick_params(labelbottom=False)
        ax9 = plt.subplot(n_params, 3, 12)
        ax9.plot(est_par[:, 3], 'g', linewidth=1)
        ax9.title.set_text('Estimated beta')
        plt.grid(True)
        plt.show()

    else:
        X = np.load(go_up(1) + '/saved_data/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        for day in range(100):
            start = time.time()
            for stock in range(n_stocks):
                x = X[day, stock, :]
                init_params = np.random.uniform(0, 1, size=4)
                b, a, xi = estimation(
                    reduced_loglikelihood.complete_loglikelihood, x, init_params, targeting_estimation=args.targ_est, verbose=args.verbose, visualization=args.visualization)
                if (math.isnan(b[-1])) or (b[-1] < 0):
                    print(b[-1])
            end = time.time()
            print(f'time day {day}: ', end - start)
