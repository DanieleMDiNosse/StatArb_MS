import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from gas import ML_errors, b_error, likelihood_hess, likelihood_jac
from regression_parameters import regression
from scipy.optimize import minimize
from scipy.stats import chi2, norm, poisson
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def synt_data(model, *args, dynamics, size):
    np.random.seed(666)
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)

    if model == 'autoregressive':
        X[1] = np.random.normal(0, sgm)
        for t in range(1, size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * X[t - 1] * \
                    (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / int(size / 10))
            if dynamics == 'step':
                b[:2] = 0.1
                if (t < 300):
                    b[t + 1] = 0.1
                if (t >= 300):
                    b[t + 1] = 0.5
                if (t >= 600):
                    b[t + 1] = 0.1
            X[t + 1] = a + b[t + 1] * X[t] + np.random.normal(0, sgm)

    if model == 'poisson':
        X[0] = np.random.poisson(b[0])
        for t in range(size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * \
                    (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 + 0.5 * np.sin(np.pi * (t + 1) / int(size / 5))
            if dynamics == 'step':
                if t < 300:
                    b[t + 1] = 0.1
                if t >= 300:
                    b[t + 1] = 0.5
                if t >= 600:
                    b[t + 1] = 0.1
            X[t + 1] = np.random.poisson(np.exp(b[t + 1]))
    return X, b

def model_loglikelihood(params, X, model):
    T = X.shape[0]
    b = np.zeros_like(X)
    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = params
        for i in range(1, T - 1):
            b[i + 1] = omega + alpha * \
                (X[i] - a - b[i] * X[i - 1]) * X[i - 1] / sgm**2 + beta * b[i]
            # b[i + 1] = omega + alpha * X[i - 1] * np.exp(-b[i]) / (sgm**2 * (1 + np.exp(-b[i]))**2) * (X[i] - X[i - 1] / (1 + np.exp(-b[i]))) + beta * b[i]

        sum = 0
        for i in range(T - 1):
            sum += - 0.5 * np.log(2 * np.pi * sgm**2) - 0.5 / \
                sgm**2 * (X[i + 1] - a - b[i + 1] * X[i])**2
            # sum += - 0.5 * np.log(2*np.pi*sgm**2) - 0.5 / sgm**2 * (X[i + 1] - X[i] / (1 + np.exp(-b[i + 1])))**2

    if model == 'poisson':
        alpha, beta, omega = params
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        sum = 0
        for t in range(T):
            sum = sum + poisson.logpmf(X[t], mu=np.exp(b[t]))

    return - sum / T


def model_estimation(X, model, specification, method='BFGS'):
    if model == 'autoregressive':
        num_par = 5
    if model == 'poisson':
        num_par = 3

    init_params = np.random.uniform(0, 1, size=num_par)
    res = minimize(model_loglikelihood, init_params, (X, model), method=method)

    while res.success == False:
        init_params = np.random.uniform(0, 1, size=num_par)
        res = minimize(model_loglikelihood, init_params,
                       (X, model), method=method)

    estimates = res.x
    if method != 'Nelder-Mead':
        std_err = ML_errors(res.jac, res.hess_inv, estimates, X, specification)
    else:
        print('Standard errors on parameters can not be computed directly from minimize output due to the non-gradient nature of the optimizer')
        std_err = [0, 0, 0, 0]

    T = X.shape[0]
    b = np.zeros_like(X)
    dbda, dbdw, dbdal, dbdb, dbds, dfidb = np.zeros_like(X), np.zeros_like(
        X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = estimates

        for t in range(1, T - 1):
            b[t + 1] = omega + alpha * X[t - 1] * \
                (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
            # b[t + 1] = omega + alpha * X[t - 1] * np.exp(-b[t]) / (sgm**2 * (1 + np.exp(-b[t]))**2) * (X[t] - X[t - 1] / (1 + np.exp(-b[t]))) + beta * b[t]
            dbda[t + 1] = - alpha * X[t - 1] / \
                sgm**2 * (1 + X[t - 1] * dbda[t]) + beta * dbda[t]
            dbdw[t + 1] = 1 - alpha / sgm**2 * \
                X[t - 1]**2 * dbdw[t] + beta * dbdw[t]
            dbdal[t + 1] = X[t - 1] / sgm**2 * \
                (X[t] - a - b[t] * X[t - 1] - alpha *
                 X[t - 1] * dbdal[t]) + beta * dbdal[t]
            dbdb[t + 1] = b[t] - alpha / sgm**2 * \
                X[t - 1]**2 * dbdb[t] + beta * dbdb[t]
            dbds[t + 1] = alpha * X[t - 1] / sgm**2 * \
                (2 / (sgm) * (b[t] * X[t - 1] - X[t] + a) -
                 X[t - 1] * dbds[t]) + beta * dbds[t]

        delta_a = beta * dbda[-1] - alpha / sgm**2 * X[-2]
        delta_w = beta * dbdw[-1] + 1
        delta_al = beta * dbdal[-1] + 1 / sgm**2 * \
            (X[-1] - a - b[-1] * X[-2]) * X[-2]
        delta_b = beta * dbdb[-1] + b[-1]
        delta_s = beta * dbds[-1] - 2 * alpha / \
            (sgm**3) * (X[-1] - a - b[-1] * X[-2]) * X[-2]

        deltas = np.array([delta_a, delta_w, delta_al, delta_b, delta_s])
        # deltas = np.zeros(shape=5)

    if model == 'poisson':
        alpha, beta, omega = estimates

        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

            dfidb[t + 1] = beta + alpha * \
                np.exp(b[t]) * (X[t] - 2 * np.exp(b[t]))
            dbdw[t + 1] = dbdw[t] * (alpha * X[t - 1] * np.exp(b[t - 1]) -
                                     2 * alpha * np.exp(2 * b[t - 1]) + beta) + 1
            dbdal[t + 1] = dbdal[t] * (alpha * X[t - 1] * np.exp(b[t - 1]) - 2 * alpha * np.exp(
                b[t - 1]) + beta) + np.exp(b[t - 1]) * (X[t - 1] - np.exp(b[t - 1]))
            dbdb[t + 1] = dbdb[t] * (alpha * X[t - 1] * np.exp(b[t - 1]) -
                                     2 * alpha * np.exp(2 * b[t - 1]) + beta) + b[t - 1]

        delta_w = dfidb * dbdw + 1
        delta_al = dfidb * dbdal + (X - np.exp(b)) * np.exp(b)
        delta_b = dfidb * dbdb + b
        deltas = np.array([delta_w[-1], delta_al[-1], delta_b[-1]])

    if method != 'Nelder-Mead':
        std_b = b_error(res.jac, res.hess_inv, deltas, X, specification)
        # std_b = 0
    else:
        print('Standard errors on b can not be computed directly from minimize output due to the non-gradient nature of the optimizer')
        std_b = 0

    return b, res, std_err, std_b, init_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-c", "--convergence", action='store_true')
    parser.add_argument("-m", "--model", type=int,
                        help='0 for AR(1), 1 for Poisson')
    parser.add_argument("-d", "--dynamics", type=int,
                        help='Dynamics for b: 0 for GAS, 1 for sinusodial, 2 for step function, 3 for exponential decay')
    parser.add_argument("-lm", "--lmtest", action='store_true', help='Lm test')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    plt.style.use('seaborn')

    if args.model == 0:
        model = 'autoregressive'
    if args.model == 1:
        model = 'poisson'

    if args.dynamics == 0:
        dynamics = 'gas'
        specification = 'correct'
    if args.dynamics == 1:
        dynamics = 'sin'
        specification = 'mis'
    if args.dynamics == 2:
        dynamics = 'step'
        specification = 'mis'

    logging.info(f' Model: {model}')
    logging.info(f' Dynamics of b: {dynamics}')
    logging.info(f' Specification: {specification}')
    time.sleep(1)

    n = 1000
    if model == 'autoregressive':
        omega = 0.05
        alpha = 0.08
        beta = 0.06
        sgm = 0.1
        a = 0.1
        X, b = synt_data(model, a, omega, alpha,
                         beta, sgm, dynamics=dynamics, size=n)
    if model == 'poisson':
        alpha = 0.081
        beta = -0.395
        omega = 0.183
        X, b = synt_data(model, alpha, beta, omega, dynamics=dynamics, size=n)

    else:
        np.random.seed()
        fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
        axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
        axs[1].plot(b[:n], 'k', label='Real', linewidth=1)
        axs[0].set_ylabel('X')
        axs[1].set_ylabel('b')

        B, res, std_err, std_b, init_params = model_estimation(
            X, model, specification, method='BFGS')
        print(res)

        # MSE = mean_squared_error(b, B)
        # print(f'MSE: {MSE}')
        ### Targeting estimation ###
        # b_bar = w / (1 - beta) -> (1 - beta) * b_bar = w

        if model == 'autoregressive':
            print('True values: ', [a, omega, alpha, beta, sgm])
        if model == 'poisson':
            print('True values: ', [alpha, beta, omega])
        print('Initial values: ', init_params)
        print('Estimated values: ', res.x)
        print('Standard errors: ', std_err)
        print('Error on b: ', std_b)

        axs[1].plot(B[:n], 'crimson', label='Filtered', linewidth=1)
        axs[1].hlines(B[:n].mean(), 0, n, 'crimson', label='Filtered mean: {:.2f}'.format(
            B.mean()), linestyle='dashed', linewidth=1)
        axs[1].fill_between(list(range(n)), B[:n] + 2 * std_b,
                            B[:n] - 2 * std_b, color='crimson', label=r'$2\sigma_b: {:.2f}$'.format(2 * std_b), alpha=0.4)

        axs[0].legend()
        axs[1].legend()
        plt.show()

    if args.lmtest:
        params = [res.x[0], res.x[1], res.x[4]]
        pvalue = LM_test_statistic(X, params)
        print('pvalue', pvalue)
        exit()

    if args.convergence:
        # N = 100
        # NN = 1
        # c = 0
        # sgmb_list = []
        # for i in tqdm(range(NN)):
        #     # np.random.seed()
        #     b_list = np.empty(shape=N)
        #     X, b = synt_data(model, a, omega, alpha,
        #                      beta, sgm, dynamics=dynamics, size=n)
        #     for j in range(N):
        #         B, res, std_err, deltas, init_params = model_estimation(X, model, specification)
        #         b_list[j] = B[-1]
        #     scaler = MinMaxScaler()
        #     b_list = scaler.fit_transform(b_list.reshape(-1,1))
        #     sgmb_list.append(b_list.std())
        # print(sgmb_list)
        # # plt.figure(figsize=(8, 6), tight_layout=True)
        # # plt.hist(sgmb_list, bins=40, color='blue', edgecolor='darkblue', alpha=0.6)
        # # plt.title(r"$b_{T}$ standard deviations")
        # # plt.savefig('../b_1000data_100init.png')
        # # plt.show()

        if model == 'autoregressive':
            num_par = 5
        if model == 'poisson':
            num_par = 3
        N = 100
        est_par = np.empty(shape=(N, num_par))
        stderr_par = np.empty(shape=(N, num_par))
        par = np.random.uniform(0, 1, size=(N, num_par))
        fun_val = []
        b_val = []
        for i in tqdm(range(len(par))):
            B, res, std_err, std_b, init_params = model_estimation(
                X, model, specification)
            fun_val.append(res.fun)
            est_par[i] = res.x
            stderr_par[i] = 2 * std_err
            b_val.append(B[-1])
        plt.figure(figsize=(12, 8))
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'crimson', linewidth=1)
        ax0.errorbar(np.arange(
            0, N), b_val, yerr=std_b, elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax0.title.set_text(r'$b_{60}$')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'crimson', linewidth=1)
        ax1.title.set_text('Likelihood')
        ax2 = plt.subplot(num_par, 3, 2)
        ax2.plot(par[:, 0], 'slateblue', linewidth=1)
        ax2.title.set_text(r'$\omega_{init}$')
        ax2.tick_params(labelbottom=False)
        ax3 = plt.subplot(num_par, 3, 5)
        ax3.plot(par[:, 1], 'slateblue', linewidth=1)
        ax3.title.set_text(r'$a_{init}$')
        ax3.tick_params(labelbottom=False)
        ax4 = plt.subplot(num_par, 3, 8)
        ax4.plot(par[:, 2], 'slateblue', linewidth=1)
        ax4.title.set_text(r'$\alpha_{init}$')
        ax4.tick_params(labelbottom=False)
        ax5 = plt.subplot(num_par, 3, 11)
        ax5.plot(par[:, 3], 'slateblue', linewidth=1)
        ax5.title.set_text(r'$\beta_{init}$')
        ax5.tick_params(labelbottom=False)
        ax11 = plt.subplot(num_par, 3, 14)
        ax11.plot(par[:, 4], 'slateblue', linewidth=1)
        ax11.title.set_text(r'$\sigma_{init}$')

        ax6 = plt.subplot(num_par, 3, 3)
        ax6.plot(est_par[:, 0], 'g', linewidth=1)
        ax6.errorbar(np.arange(
            0, N), est_par[:, 0], yerr=stderr_par[:, 0], elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax6.hlines(omega, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax6.title.set_text(r'$\hat{\omega}$')
        ax6.tick_params(labelbottom=False)
        ax7 = plt.subplot(num_par, 3, 6)
        ax7.plot(est_par[:, 1], 'green', linewidth=1)
        ax7.errorbar(np.arange(
            0, N), est_par[:, 1], yerr=stderr_par[:, 1], elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax7.hlines(a, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax7.title.set_text(r'$\hat{a}$')
        ax7.tick_params(labelbottom=False)
        ax8 = plt.subplot(num_par, 3, 9)
        ax8.plot(est_par[:, 2], 'green', linewidth=1)
        ax8.errorbar(np.arange(
            0, N), est_par[:, 2], yerr=stderr_par[:, 2], elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax8.hlines(alpha, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax8.title.set_text(r'$\hat{\alpha}$')
        ax8.tick_params(labelbottom=False)
        ax9 = plt.subplot(num_par, 3, 12)
        ax9.plot(est_par[:, 3], 'green', linewidth=1)
        ax9.errorbar(np.arange(
            0, N), est_par[:, 3], yerr=stderr_par[:, 3], elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax9.hlines(beta, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax9.title.set_text(r'$\hat{\beta}$')
        ax9.tick_params(labelbottom=False)
        ax10 = plt.subplot(num_par, 3, 15)
        ax10.plot(est_par[:, 4], 'green', linewidth=1)
        ax10.errorbar(np.arange(
            0, N), est_par[:, 4], yerr=stderr_par[:, 4], elinewidth=0.6, capsize=3, capthick=1, errorevery=int(N / 4))
        ax10.hlines(sgm, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax10.title.set_text(r'$\hat{\sigma}$')
        plt.grid(True)
        plt.show()
