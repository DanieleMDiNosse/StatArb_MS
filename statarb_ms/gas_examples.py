from scipy.optimize import minimize
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time
from regression_parameters import regression
from gas import ML_errors, likelihood_jac, likelihood_hess, b_error


def synt_data(model, *args, dynamics, size):
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)

    if model == 'autoregressive':
        eps = np.random.normal(0, sgm, size=size)
        X[1] = eps[1]
        for t in range(1, X.shape[0] - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * X[t - 1] * \
                    (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / 100)
            if dynamics == 'step':
                b[:2] = 0.1
                if (t < 300):
                    b[t + 1] = 0.1
                if (t >= 300):
                    b[t + 1] = 0.5
                if (t >= 600):
                    b[t + 1] = 0.1
            if dynamics == 'ar1':
                w, k = 0.1, 0.5
                b[t + 1] = w + k * b[t] + np.random.normal(0, 0.1)
            X[t + 1] = a + b[t + 1] * X[t] + eps[t + 1]

    if model == 'poisson':
        X[0] = np.random.poisson(b[0])
        for t in range(size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * \
                    (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / 150)
            if dynamics == 'step':
                if t < 300:
                    b[t + 1] = 0.1
                if t >= 300:
                    b[t + 1] = 1
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

        sum = 0
        for i in range(1, T - 1):
            sum += (- 0.5 * np.log(2*np.pi*sgm**2) - 0.5 *
                    (X[i + 1] - a - b[i + 1] * X[i])**2 / sgm**2)

    if model == 'poisson':
        alpha, beta, omega = params
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        sum = 0
        for t in range(T):
            sum = sum + poisson.logpmf(X[t], mu=np.exp(b[t]))

    return - sum / T


def model_estimation(fun, X, init_params, model, specification):
    res = minimize(fun, init_params, (X, model), method='BFGS',
                   options={'maxiter': 1000})
    # res = minimize(fun, init_params, (X, model), method='dogleg', jac=likelihood_jac, hess=likelihood_hess,
    #                options={'maxiter': 1000})
    estimates = res.x
    std_err = ML_errors(res.jac, res.hess_inv, estimates, X, specification)

    T = X.shape[0]
    b = np.zeros_like(X)
    dbda, dbdw, dbdal, dbdb, dbds, dfidb = np.zeros_like(X), np.zeros_like(
        X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = estimates
        for t in range(1, T - 1):
            b[t + 1] = omega + alpha * X[t - 1] * \
                (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]

            dbda[t + 1] = - alpha * X[t - 1] / \
                sgm**2 * (1 + X[t - 1] * dbda[t]) + beta * dbda[t]
            dbdw[t + 1] = 1 - alpha / sgm**2 * X[t - 1]**2 * dbdw[t] + beta * dbdw[t]
            dbdal[t + 1] = X[t - 1] / sgm**2 * \
                (X[t] - a - b[t] * X[t - 1] - alpha * X[t - 1] * dbdal[t]) + beta * dbdal[t]
            dbdb[t + 1] = b[t] - alpha / sgm**2 * X[t - 1]**2 * dbdb[t] + beta * dbdb[t]
            dbds[t + 1] = alpha * X[t - 1] / sgm**2 * \
                (2 / (sgm) * (b[t] * X[t - 1] - X[t] + a) - X[t - 1] * dbds[t]) + beta * dbds[t]

        delta_a = beta * dbda[-1] - alpha / sgm**2 * X[-2]
        delta_w = beta * dbdw[-1] + 1
        delta_al = beta * dbdal[-1] + 1 / sgm**2 * \
            (X[-1] - a - b[-1] * X[-2]) * X[-2]
        delta_b = beta * dbdb[-1] + b[-1]
        delta_s = beta * dbds[-1] - 2 * alpha / \
            (sgm**3) * (X[-1] - a - b[-1] * X[-2]) * X[-2]

        deltas =  np.array([delta_a, delta_w, delta_al, delta_b, delta_s])

    # jac = likelihood_jac(estimates, X, b, model)
    # hess_inv = np.linalg.inv(likelihood_hess(estimates, X, b, model))
    # std_err = ML_errors(jac, hess_inv, estimates, X, specification)

    if model == 'poisson':
        alpha, beta, omega = estimates
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

            dfidb[t + 1] = beta + alpha * np.exp(b[t]) * (X[t] - 2 * np.exp(b[t]))
            dbdw[t + 1] = 1 + alpha * X[t - 1] * np.exp(b[t - 1]) * dbdw[t] - 2* alpha * np.exp(2*b[t - 1]) * dbdw[t] + beta * dbdw[t]
            dbdal[t + 1] = X[t - 1] * np.exp(b[t - 1]) + alpha * X[t - 1] * np.exp(b[t - 1]) * dbdal[t] - np.exp(2*b[t - 1]) + 2*alpha * np.exp(b[t - 1]) * dbdal[t] + beta * dbdal[t]
            dbdb[t + 1] = alpha * X[t - 1] * np.exp(b[t - 1]) * dbdb[t] - 2 * alpha * np.exp(b[t- 1]) + b[t - 1] + beta * dbdb[t]

        delta_w = dfidb * dbdw + 1
        delta_al = dfidb * dbdal + (X - np.exp(b) * np.exp(b))
        delta_b = dfidb * dbdb + b
        deltas =  np.array([delta_w[-1], delta_al[-1], delta_b[-1]])

    std_b = b_error(res.jac, res.hess_inv, deltas, X, specification)

    return b, res, std_err, std_b


def LM_test_statistic(Xr, params):
    omega, a, sigma = params[0], params[1], params[2]
    T = Xr.shape[0]
    Y = np.zeros_like(Xr)
    X = np.zeros_like(Xr)
    for t in range(1, T):
        Y[t] = 1 / (Xr[t - 1] / sigma**2 * (Xr[t] - omega * Xr[t - 1] - a))
    for t in range(2, T):
        X[t] = Xr[t - 1] / sigma**2 * (Xr[t] - omega * Xr[t - 1] - a)
    c_w, c_a, conf_intervals, residuals, predictions, rsquared = regression(
        X, Y)
    print(c_w, c_a)

    Y_est = c_w + c_a * X[t]
    ESS = ((Y_est - Y.mean())**2).sum()

    return ESS


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
    plt.style.use('seaborn')
    # np.random.seed(666)

    if args.model == 0:
        model = 'autoregressive'
    if args.model == 1:
        model = 'poisson'
    if args.dynamics == 0:
        dynamics = 'gas'
        specification = 'well'
    if args.dynamics == 1:
        dynamics = 'sin'
        specification = 'mis'
    if args.dynamics == 2:
        dynamics = 'step'
        specification = 'mis'
    if args.dynamics == 3:
        dynamics = 'ar1'
        specification = 'mis'

    n = 1000
    if model == 'autoregressive':
        num_par = 5
        omega = 0.05
        alpha = 0.08
        beta = 0.06
        sgm = 0.1
        a = 0.1
        X, b = synt_data(model, a, omega, alpha,
                         beta, sgm, dynamics=dynamics, size=n)
    if model == 'poisson':
        num_par = 3
        alpha = 0.081
        beta = -0.395
        omega = 0.183
        X, b = synt_data(model, alpha, beta, omega, dynamics=dynamics, size=n)

    if args.lmtest:
        params = [omega, a, sgm]
        ESS = LM_test_statistic(X, params)
        print(ESS)
        time.sleep(10)

    else:
        init_params = np.random.uniform(0, 0.5, size=num_par)
        fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
        axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
        axs[1].plot(b[:n], 'k', label='Real', linewidth=1)
        axs[0].set_ylabel('X')
        axs[1].set_ylabel('b')

        B, res, std_err, std_b = model_estimation(
            model_loglikelihood, X, init_params, model, specification)
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
        # axs[1].plot(b_test[:n], 'blue', label='Manual', linewidth=1)
        axs[1].fill_between(list(range(n)), B[:n] + 2 * std_b,
                            B[:n] - 2 * std_b, color='crimson', label=r'$2\sigma_b: {:.2f}$'.format(2*std_b), alpha=0.4)

        axs[0].legend()
        axs[1].legend()
        plt.show()

    if args.convergence:
        N = 20
        est_par = np.empty(shape=(N, num_par))
        stderr_par = np.empty(shape=(N, num_par))
        par = np.random.uniform(0, 0.5, size=(N, num_par))
        fun_val = []
        b_val = []
        for i in tqdm(range(len(par))):
            B, res, std_err, deltas = model_estimation(
                model_loglikelihood, X, par[i], model, specification)
            fun_val.append(res.fun)
            est_par[i] = res.x
            stderr_par[i] = 2*std_err
            b_val.append(B[-1])
        plt.figure(figsize=(12, 8))
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'crimson', linewidth=1)
        ax0.title.set_text(r'$b_{60} values$')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'crimson', linewidth=1)
        ax1.title.set_text('Likelihood Evaluations')
        ax2 = plt.subplot(num_par, 3, 2)
        ax2.plot(par[:, 0], 'slateblue', linewidth=1)
        ax2.title.set_text(r'$Initial \omega$')
        ax2.tick_params(labelbottom=False)
        ax3 = plt.subplot(num_par, 3, 5)
        ax3.plot(par[:, 1], 'slateblue', linewidth=1)
        ax3.title.set_text('Initial a')
        ax3.tick_params(labelbottom=False)
        ax4 = plt.subplot(num_par, 3, 8)
        ax4.plot(par[:, 2], 'slateblue', linewidth=1)
        ax4.title.set_text(r'$Initial \alpha$')
        ax4.tick_params(labelbottom=False)
        ax5 = plt.subplot(num_par, 3, 11)
        ax5.plot(par[:, 3], 'slateblue', linewidth=1)
        ax5.title.set_text(r'$Initial \beta$')
        ax5.tick_params(labelbottom=False)
        ax11 = plt.subplot(num_par, 3, 14)
        ax11.plot(par[:, 4], 'slateblue', linewidth=1)
        ax11.title.set_text(r'$Initial \sigma$')

        ax6 = plt.subplot(num_par, 3, 3)
        ax6.plot(est_par[:, 0], 'g', linewidth=1)
        ax6.errorbar(np.arange(
            0, N), est_par[:, 0], yerr=stderr_par[:, 0], elinewidth=0.6, capsize=3, capthick=1)
        ax6.hlines(omega, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax6.title.set_text(r'$Estimated \omega$')
        ax6.tick_params(labelbottom=False)
        ax7 = plt.subplot(num_par, 3, 6)
        ax7.plot(est_par[:, 1], 'green', linewidth=1)
        ax7.errorbar(np.arange(
            0, N), est_par[:, 1], yerr=stderr_par[:, 1], elinewidth=0.6, capsize=3, capthick=1)
        ax7.hlines(a, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax7.title.set_text('Estimated a')
        ax7.tick_params(labelbottom=False)
        ax8 = plt.subplot(num_par, 3, 9)
        ax8.plot(est_par[:, 2], 'green', linewidth=1)
        ax8.errorbar(np.arange(
            0, N), est_par[:, 2], yerr=stderr_par[:, 2], elinewidth=0.6, capsize=3, capthick=1)
        ax8.hlines(alpha, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax8.title.set_text(r'$Estimated \alpha$')
        ax8.tick_params(labelbottom=False)
        ax9 = plt.subplot(num_par, 3, 12)
        ax9.plot(est_par[:, 3], 'green', linewidth=1)
        ax9.errorbar(np.arange(
            0, N), est_par[:, 3], yerr=stderr_par[:, 3], elinewidth=0.6, capsize=3, capthick=1)
        ax9.hlines(beta, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax9.title.set_text(r'$Estimated \beta$')
        ax9.tick_params(labelbottom=False)
        ax10 = plt.subplot(num_par, 3, 15)
        ax10.plot(est_par[:, 4], 'green', linewidth=1)
        ax10.errorbar(np.arange(
            0, N), est_par[:, 4], yerr=stderr_par[:, 4], elinewidth=0.6, capsize=3, capthick=1)
        ax10.hlines(sgm, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax10.title.set_text(r'$Estimated \sigma$')
        plt.grid(True)
        plt.show()
