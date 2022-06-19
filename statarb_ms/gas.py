import argparse
import math
import time

from StatArb_MS.statarb_ms import loglikelihood
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm


def ML_errors(jac, hess_inv, params, X, specification):
    T = X.shape[0]
    num_par = params.shape[0]
    J = np.outer(jac, jac) / T

    if specification == 'mis':
        var = np.dot(np.dot(hess_inv, J), hess_inv)
        std_err = np.sqrt([var[i, i] / T for i in range(num_par)])
    if specification == 'correct':
        std_err = np.sqrt([hess_inv[i, i] / T for i in range(num_par)])

    return std_err


def b_error(X, res, M, model, link_fun, specification):
    T = X.shape[0]
    b = np.zeros(shape=(M, T))
    J = np.outer(res.jac, res.jac)
    estimates = res.x

    for m in range(M):
        if model == 'autoregressive':
            if specification == 'correct':
                a, omega, alpha, beta, sgm = np.random.multivariate_normal(
                    estimates, res.hess_inv / T)

                for t in range(1, T - 1):
                    if link_fun == 'identity':
                        b[m, t + 1] = omega + alpha * X[t - 1] * \
                            (X[t] - a - b[m, t] * X[t - 1]) / \
                            sgm**2 + beta * b[m, t]

                    if link_fun == 'logistic':
                        b[m, t + 1] = omega + alpha * X[t - 1] * \
                            (X[t] - a - 1 / (1 + np.exp(-b[m, t])) * X[t - 1]) / \
                            sgm**2 + beta * b[m, t]

                    if link_fun == 'identity_student':
                        b[m, t + 1] = omega + alpha * (lam + 1) * X[t - 1] * (X[t] - a - b[m, t] * X[t - 1])  / (lam + (X[t] - a - b[m, t] * X[t - 1])**2) + beta * b[m, t]

            else:
                a, omega, alpha, beta, sgm = np.random.multivariate_normal(
                    estimates, np.dot(np.dot(res.hess_inv, J), res.hess_inv) / T)

                for t in range(1, T - 1):
                    if link_fun == 'identity':
                        b[m, t + 1] = omega + alpha * X[t - 1] * \
                            (X[t] - a - b[m, t] * X[t - 1]) / \
                            sgm**2 + beta * b[m, t]

                    if link_fun == 'logistic':
                        b[m, t + 1] = omega + alpha * X[t - 1] * \
                            (X[t] - a - 1 / (1 + np.exp(-b[m, t])) * X[t - 1]) / \
                            sgm**2 + beta * b[m, t]

                    if link_fun == 'identity_student':
                        b[m, t + 1] = omega + alpha * (lam + 1) * X[t - 1] * (X[t] - a - b[m, t] * X[t - 1])  / (lam + (X[t] - a - b[m, t] * X[t - 1])**2) + beta * b[m, t]

        if model == 'poisson':
            if specification == 'correct':
                alpha, beta, omega = np.random.multivariate_normal(
                    estimates, res.hess_inv / T)

                for t in range(T - 1):
                    b[m, t + 1] = omega + alpha * \
                        (X[t] - np.exp(b[m, t])) * \
                        (np.exp(b[m, t])) + beta * b[m, t]

            else:
                alpha, beta, omega = np.random.multivariate_normal(
                    estimates, np.dot(np.dot(res.hess_inv, J), res.hess_inv) / T)

                for t in range(T - 1):
                    b[m, t + 1] = omega + alpha * \
                        (X[t] - np.exp(b[m, t])) * \
                        (np.exp(b[m, t])) + beta * b[m, t]

    std_b = b.std(axis=0)

    return std_b


def estimation(X, n_iter, link_fun, targeting_estimation=False, verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.zeros(shape=T)
    xi = np.zeros(shape=T)

    if targeting_estimation:
        pass
        # res_iter = np.zeros(shape=(n_iter,4))
        # for i in range(1):
        #     init_params = np.random.uniform(0, 1, size=2)
        #     res = minimize(loglikelihood.targeting_loglikelihood, init_params, X, method='Nelder-Mead')
        #
        # omega, a = res.x
        # b_bar = omega
        # b[:2] = b_bar
        #
        # for i in range(res_iter.shape[0]):
        #      init_params = np.random.uniform(0, 1, size=3)
        #      sgm = 1
        #      res = minimize(loglikelihood.loglikelihood_after_targ, init_params, (X,b_bar),  method='Nelder-Mead')
        #      if np.isnan(res.fun) == False:
        #          res_iter[i, :-1] = res.x
        #          res_iter[i, -1] = res.fun
        #
        # a, alpha, beta = res.x[0], res.x[1], res.x[2]
        # omega = beta * (1 - b_bar)
        # res = [omega, a, alpha, beta]

    else:
        res_iter = np.zeros(shape=(n_iter, 5))

        for i in range(res_iter.shape[0]):
            init_params = np.random.uniform(0, 1, size=4)
            res = minimize(loglikelihood.loglikelihood,
                           init_params, (X, link_fun), method='Nelder-Mead')

            while np.isnan(res.fun) == True:
                init_params = np.random.uniform(0, 1, size=4)
                res = minimize(loglikelihood.loglikelihood,
                               init_params, (X, link_fun), method='Nelder-Mead')

            res_iter[i, :-1] = res.x
            res_iter[i, -1] = res.fun

            # if np.isnan(res.fun) == False:
            #     res_iter[i, :-1] = res.x
            #     res_iter[i, -1] = res.fun

        init_params = res_iter[np.where(
            res_iter[:, 4] == res_iter[:, 4].min())][0][:4]

        res = minimize(loglikelihood.loglikelihood,
                       init_params, (X, link_fun), method='Nelder-Mead')
        res = minimize(loglikelihood.loglikelihood,
                       res.x, (X, link_fun), method='BFGS', options={'maxiter': 1})

        omega, a, alpha, beta = res.x
        res = [omega, a, alpha, beta]
        sgm, lam = 1, 1

    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(2.5)

    for t in range(1, T - 1):
        if link_fun == 'identity':
            b[t + 1] = omega + alpha * xi[t] * X[t - 1] / sgm**2 + beta * b[t]
            xi[t + 1] = X[t - 1] - a - b[t - 1] * X[t]

        if link_fun == 'logistic':
            b[t + 1] = omega + alpha * (X[t] - a - 1 / (1 + np.exp(-b[t])) *
                                        X[t - 1]) * np.exp(-b[t]) / (1 + np.exp(-b[t]))**2 * X[t - 1] / sgm**2 + beta * b[t]
            xi[t + 1] = X[t + 1] - a - 1 / (1 + np.exp(-b[t + 1])) * X[t]

        if link_fun == 'identity_student':
            b[t + 1] = omega + alpha * (lam + 1) * X[t - 1] * (X[t] - a - b[t] * X[t - 1]) / (lam + (X[t] - a - b[t] * X[t - 1])**2) + beta * b[t]
            xi[t + 1] = X[t + 1] - a - b[t + 1] * X[t]


    if visualization:
        plt.figure(figsize=(12, 5), tight_layout=True)
        plt.plot(b, linewidth=1, label='Filtered Data')
        plt.legend()
        plt.grid(True)
        plt.title(r'$[\omega, a, \alpha, \beta]: {:.4f} , {:.4f} , {:.4f} , {:.4f} $'.format(
            res.x[0],  res.x[1],  res.x[2], res.x[3]))
        plt.show()

    if link_fun == 'logistic': b = 1 / (1 + np.exp(-b))

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
    # np.random.seed(666)

    if args.update == 0:
        update = 'gaussian'
    if args.update == 1:
        update = 'logistic'

    tickers = pd.read_pickle('/mnt/saved_data/ReturnsData.pkl').columns
    name = input('Name of the file of residuals: ')
    X = np.load(f'/mnt/saved_data/{name}.npy')

    if args.convergence:
        n_params = 4
        N = 10000
        days = X.shape[0]
        n_stocks = X.shape[1]
        day = np.random.randint(0, days)
        stock = np.random.randint(0, n_stocks)
        day = 1592
        stock = tickers.get_loc('JWN')
        # print(f'Day: {day}, Stock: {tickers[stock]}')
        x = X[day, stock, :]
        results = np.empty(shape=(N, n_params))
        for n_iter in [2, 7, 9, 14]:
            for j in tqdm(range(N), desc=f'{n_iter}'):
                b, a, xi, res = estimation(
                    x, n_iter, targeting_estimation=args.targ_est)
                results[j] = res
                omega, a, alpha, beta = results[:,
                                                0], results[:, 1], results[:, 2], results[:, 3]

            plt.figure(figsize=(12, 8), tight_layout=True)
            ax1 = plt.subplot(2, 2, 1)
            ax1.hist(omega, bins=100, color='blue', alpha=0.6)
            ax1.title.set_text(r'$\hat{\omega}$')
            ax2 = plt.subplot(2, 2, 2)
            ax2.hist(a, bins=100, color='blue', alpha=0.6)
            ax2.title.set_text(r'$\hat{a}$')
            ax3 = plt.subplot(2, 2, 3)
            ax3.hist(alpha, bins=100, color='blue', alpha=0.6)
            ax3.title.set_text(r'$\hat{\alpha}$')
            ax4 = plt.subplot(2, 2, 4)
            ax4.hist(beta, bins=100, color='blue', alpha=0.6)
            ax4.title.set_text(r'$\hat{\beta}$')
            plt.savefig(
                f'../../convergence/Day:_{day},_Stock:_JWM_,_Res:_dis_res80_{n_iter+1}NM.png')
        plt.show()

        # plt.figure(figsize=(12, 8), tight_layout=True)
        # ax0 = plt.subplot(2, 3, 4)
        # ax0.plot(s_val, 'crimson', linewidth=1)
        # ax0.title.set_text('s-score')
        # ax1 = plt.subplot(2, 3, 1)
        # ax1.plot(fun_val, 'crimson', linewidth=1)
        # ax1.title.set_text('Likelihood Evaluations')
        # ax2 = plt.subplot(n_params, 3, 2)
        # ax2.plot(par[:, 0], 'slateblue', linewidth=1)
        # ax2.title.set_text('Initial Omega')
        # ax2.tick_params(labelbottom=False)
        # ax3 = plt.subplot(n_params, 3, 5)
        # ax3.plot(par[:, 1], 'slateblue', linewidth=1)
        # ax3.title.set_text('Initial a')
        # ax3.tick_params(labelbottom=False)
        # ax4 = plt.subplot(n_params, 3, 8)
        # ax4.plot(par[:, 2], 'slateblue', linewidth=1)
        # ax4.title.set_text('Initial alpha')
        # ax4.tick_params(labelbottom=False)
        # ax5 = plt.subplot(n_params, 3, 11)
        # ax5.plot(par[:, 3], 'slateblue', linewidth=1)
        # ax5.title.set_text('Initial beta')
        #
        # ax6 = plt.subplot(n_params, 3, 3)
        # ax6.plot(est_par[:, 0], 'g', linewidth=1)
        # ax6.title.set_text('Estimated Omega')
        # ax6.tick_params(labelbottom=False)
        # ax7 = plt.subplot(n_params, 3, 6)
        # ax7.plot(est_par[:, 1], 'g', linewidth=1)
        # ax7.title.set_text('Estimated a')
        # ax7.tick_params(labelbottom=False)
        # ax8 = plt.subplot(n_params, 3, 9)
        # ax8.plot(est_par[:, 2], 'g', linewidth=1)
        # ax8.title.set_text('Estimated alpha')
        # ax8.tick_params(labelbottom=False)
        # ax9 = plt.subplot(n_params, 3, 12)
        # ax9.plot(est_par[:, 3], 'g', linewidth=1)
        # ax9.title.set_text('Estimated beta')
        # plt.grid(True)

    else:
        days = X.shape[0]
        n_stocks = X.shape[1]
        # day = np.random.randint(days)
        # stock = np.random.randint(n_stocks)
        # print(day, stock)
        x = X[3665, 251, :]
        bs = []
        for i in tqdm(range(100)):
            if args.targ_est:
                init_params = initial_value(
                    loglikelihood.loglikelihood, 100, x, 3)
                b, a, xi, res, std = estimation(loglikelihood.loglikelihood_after_targ,
                                                x, init_params, targeting_estimation=args.targ_est, visualization=True)
            else:
                # init_params = initial_value(loglikelihood.loglikelihood, 100, x, 4)
                init_params = np.random.uniform(size=4)
                b, a, xi, res = estimation(x, init_params,
                                           targeting_estimation=args.targ_est, verbose=args.verbose, visualization=False)
                bs.append(b[-1])
        print(np.array(bs).std())
        if (math.isnan(b[-1])) or (b[-1] < 0):
            print(b[-1])
