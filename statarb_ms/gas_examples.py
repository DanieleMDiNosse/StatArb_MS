import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from gas import ML_errors, b_error
from post_processing import LM_test_statistic
from regression_parameters import regression
from scipy.optimize import minimize
from scipy.stats import chi2, norm, normaltest, poisson, t
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings


def synt_data(model, dynamics, link_fun, *args, size):
    '''This function generates a variable following an AR(1) process with autoregressive parameter that varies in time in specific ways decided by the dynamics argument OR following a Poisson process whose paramter is determined by the dyanamics argument.

    Parameters
    ----------
    model : str
        'autoregressive' or 'poisson'.
    *args : variable number of floats
        Static parameters of the model.
    dynamics : str
        'gas', 'step' or 'sin' decide the dynamics of the autoregressive/poisson parameter.
    size : int
        Lenght of the resulting variable.

    Returns
    -------
    X : ndarray of shape (N)
        Synthetic generated variable.
    b : ndarray of shape (N)
        Time varying parameter.

    '''
    np.random.seed(666)
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)

    if model == 'autoregressive':
        X[1] = np.random.normal(0, sgm)
        for t in range(1, size - 1):
            if dynamics == 'gas':
                if link_fun == 'identity':
                    b[t + 1] = omega + alpha * X[t - 1] * \
                        (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
                if link_fun == 'logistic':
                    b[t + 1] = omega + alpha * (X[t] - 1 / (1 + np.exp(-b[t])) *
                                                X[t - 1]) * np.exp(-b[t]) / (1 + np.exp(-b[t]))**2 * X[t - 1] / sgm**2 + beta * b[t]
                if link_fun == 'identity_student':
                    b[t + 1] = omega + alpha * (lam + 1) * X[t - 1] * (X[t] - a - b[t] * X[t - 1])  / (lam + (X[t] - a - b[t] * X[t - 1])**2) + beta * b[t]

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

            if link_fun == 'logistic':
                X[t + 1] = a + 1 / (1 + np.exp(-b[t + 1])) * \
                    X[t] + np.random.normal(0, sgm)

            if link_fun == 'identity':
                X[t + 1] = a + b[t + 1] * X[t] + np.random.normal(0, sgm)

            if link_fun == 'identity_student':
                X[t + 1] = a + b[t + 1] * X[t] + t.rvs(lam)

    if model == 'poisson':
        X[0] = np.random.poisson(b[0])
        for t in range(size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * \
                    (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / int(size / 10))
            if dynamics == 'step':
                if t < 300:
                    b[t + 1] = 0.1
                if t >= 300:
                    b[t + 1] = 0.5
                if t >= 600:
                    b[t + 1] = 0.1
            X[t + 1] = np.random.poisson(np.exp(b[t + 1]))
    return X, b


def model_loglikelihood(params, X, model, link_fun):
    '''This function computes the total loglikelihood of the AR(1) model with GAS(1,1) filter on the autoregressive/poisson parameter. In the autoregressive case, the innovation is assumer to be normal distributed.

    Parameters
    ---------
    params : list
        Static parameters of the model.
    X : ndarray
        AR(1) of poisson variable generated by the synt_data function.
    model : str
        'autoregressive' or 'poisson'.

    Returns
    -------
    loglikelihood : ndarray of shape (X.shape[0])
        Total loglikehood with opposite sign divided by the length of the sample.
    '''

    T = X.shape[0]
    b = np.zeros_like(X)
    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = params
        for i in range(1, T - 1):

            if link_fun == 'identity':
                b[i + 1] = omega + alpha * \
                    (X[i] - a - b[i] * X[i - 1]) * X[i - 1] / sgm**2 + beta * b[i]

            if link_fun == 'logistic':
                b[i + 1] = omega + alpha * (X[i] - a - 1 / (1 + np.exp(-b[i])) *
                                            X[i - 1]) * np.exp(-b[i]) / (1 + np.exp(-b[i]))**2 * X[i - 1] / sgm**2 + beta * b[i]

        sum = 0
        for i in range(T - 1):
            sum += - 0.5 * np.log(2 * np.pi * sgm**2) - 0.5 / \
                sgm**2 * (X[i + 1] - a - 1 / (1 + np.exp(-b[i + 1])) * X[i])**2

    if model == 'poisson':
        alpha, beta, omega = params
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        sum = 0
        for t in range(T):
            sum = sum + poisson.logpmf(X[t], mu=np.exp(b[t]))

    loglikelihood = - sum / T
    return loglikelihood


def model_estimation(X, model, link_fun, specification):
    '''Compute the fixed parameters estimation of the AR(1) model with a GAS(1,1) filter of the autoregressive/poisson parameter. This function uses the output of model_loglikelihood and it is based on the minimize module of scipy.optimize.
    The specification argument modules the way errors are computed.

    Paramters
    ---------
    X : ndarray
        AR(1) of poisson variable generated by the synt_data function.
    model : str
        'autoregressive' or 'poisson'.
    link_fun : str
        'identity' or 'logistic'.
    specification : str
        'mis' for mis-specified, 'correct' for correct specified.

    Returns
    -------

    b : numpy ndarray
        Filterd dynamics of the autoregressive/poisson parameter.
    res : numpy ndarray
        Scipy.optimize.minimize results.
    xi : numpy ndarray
        Filtered residuals.
    std_err : numpy ndarray
        Errors of the static paramter.
    std_b : float
        Error on the filterd autoregressive/poisson parameter.
    init_params : list of floats
        Initial guess.
    B_T : float
        One step ahead forecast.
    '''
    if model == 'autoregressive':
        num_par = 5
    if model == 'poisson':
        num_par = 3
    res_iter = np.zeros(shape=(14, num_par + 1))

    # Metto da parte gli ultimi due valori di X per il forecasting
    X_Tm1 = X[-1]
    X = X[:-1]

    for i in range(res_iter.shape[0]):
        init_params = np.random.uniform(0, 1, size=num_par)
        res = minimize(model_loglikelihood, init_params,
                       (X, model, link_fun), method='Nelder-Mead')
        if np.isnan(res.fun) == False:
            res_iter[i, :-1] = res.x
            res_iter[i, -1] = res.fun

    init_params = res_iter[np.where(
        res_iter[:, -1] == res_iter[:, -1].min())][0][:-1]
    res = minimize(model_loglikelihood, init_params,
                   (X, model), method='Nelder-Mead')
    res = minimize(model_loglikelihood, res.x, (X, model),
                   method='BFGS', options={'maxiter': 1})

    estimates = res.x
    std_err = ML_errors(res.jac, res.hess_inv, estimates, X, specification)

    std_b = b_error(X, res, 2000, model, specification)

    T = X.shape[0]
    b = np.zeros_like(X)
    xi = np.zeros_like(b)
    dbda, dbdw, dbdal, dbdb, dbds, dfidb = np.zeros_like(X), np.zeros_like(
        X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = estimates

        for t in range(1, T - 1):

            if link_fun == 'identity':
                b[t + 1] = omega + alpha * X[t - 1] * \
                    (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
                xi[t + 1] = X[t + 1] - a - b[t + 1] * X[t]

            if link_fun == 'logistic':
                b[t + 1] = omega + alpha * (X[t] - a - 1 / (1 + np.exp(-b[t])) *
                                            X[t - 1]) * np.exp(-b[t]) / (1 + np.exp(-b[t]))**2 * X[t - 1] / sgm**2 + beta * b[t]
                xi[t + 1] = X[t + 1] - a - 1 / (1 + np.exp(-b[t + 1])) * X[t]

        if link_fun == 'identity':
            B_T = omega + alpha / sgm**2 * \
                (X[-1] - a - b[-1] * X[-2]) * X[-2] + beta * b[-1]

        if link_fun == 'logistic':
            B_T = omega + alpha * (X[-1] - a - 1 / (1 + np.exp(-b[-1])) *
                                        X[-2]) * np.exp(-b[-1]) / (1 + np.exp(-b[-1]))**2 * X[-2] / sgm**2 + beta * b[-1]

    if model == 'poisson':
        alpha, beta, omega = estimates
        xi = 0

        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        B_T = omega + alpha * (X[-1] - np.exp(b[-1])) * \
            (np.exp(b[-1])) + beta * b[-1]

    if link_fun == 'identity': b = 1 / (1 + np.exp(-b))

    return b, res, xi, std_err, std_b, init_params, B_T


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-c", "--convergence", action='store_true')
    parser.add_argument("-m", "--model", type=int,
                        help='0 for AR(1), 1 for Poisson')
    parser.add_argument("-lf", "--link_fun", type=int,
                        help='0 identity, 1 logistic')
    parser.add_argument("-d", "--dynamics", type=int,
                        help='Dynamics for b: 0 for GAS, 1 for sinusodial, 2 for step function, 3 for exponential decay')
    parser.add_argument("-lm", "--lmtest", action='store_true', help='LM test')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    warnings.filterwarnings('ignore') # Ignore runtime warnings. They are returned due to overflows or invalid operations. These situations are caused by the evaluations during the optimization procedure. They not necessarly mean that the estimation fails.
    plt.style.use('seaborn')

    if args.model == 0:
        model = 'autoregressive'
    if args.model == 1:
        model = 'poisson'

    if args.link_fun == 0:
        link_fun = 'identity'
    if args.link_fun == 1:
        link_fun = 'logistic'

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
    logging.info(f' Link function: {link_fun}')
    logging.info(f' Specification: {specification}')
    time.sleep(1)

    n = 1000

    if model == 'autoregressive':
        omega = 0.05
        alpha = 0.08
        beta = 0.06
        sgm = 0.1
        a = 0.1
        X, b = synt_data(model, dynamics, link_fun, a, omega, alpha,
                         beta, sgm, size=n)

    if model == 'poisson':
        alpha = 0.081
        beta = -0.395
        omega = 0.183
        X, b = synt_data(model, dynamics, link_fun, alpha, beta, omega, size=n)

    np.random.seed()
    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
    axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
    axs[1].plot(b[:n], 'k', label='Real', linewidth=1)
    axs[0].set_ylabel('X')
    axs[1].set_ylabel('b')

    B, res, xi, std_err, std_b, init_params, B_T = model_estimation(
        X, model, link_fun, specification)


    chi2 = np.sqrt((B_T - b[-1])**2 / std_b[-1]**2)

    MSE = mean_squared_error(b[:-1], B) # E' fino a -1 poiché gli ultimi due valori di X li uso per il forecast

    print('--------------------- Summary --------------------- \n')
    if model == 'autoregressive':
        if specification == 'correct':
            print('True values: ', [a, omega, alpha, beta, sgm])
    if model == 'poisson':
        if specification == 'correct':
            print('True values: ', [alpha, beta, omega])
    print('Estimated values: ', res.x)
    print('Standard errors (params): ', std_err)
    print('Standard errors (b(T)): ', std_b[-1])
    print(f'MSE: {MSE}')
    if model == 'autoregressive':
        print(
            f"D'Agostino and Pearson pvalue normality test: {normaltest(xi).pvalue}")
    print('True value of b(T+1) and forecasted one: ', b[-1], B_T)
    print('Chi squared between b(T+1) and its forecast: ', chi2)
    print('--------------------------------------------------- \n')

    axs[1].plot(B[:n], 'crimson', label='Filtered', linewidth=1)
    axs[1].hlines(B[:n].mean(), 0, n, 'crimson', label='Filtered mean: {:.2f}'.format(
        B.mean()), linestyle='dashed', linewidth=1)
    axs[1].fill_between(list(range(n - 1)), B[:n - 1] + 1.96 * std_b,
                        B[:n - 1] - 1.96 * std_b, color='crimson', alpha=0.4, label='95% conf int')

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
        N = 30
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
