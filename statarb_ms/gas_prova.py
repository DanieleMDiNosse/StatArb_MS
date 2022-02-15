from scipy.optimize import minimize
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from tqdm import tqdm
import time


def synt_data(model, *args, dynamics, size):
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)

    if model == 'autoregressive':
        eps = np.random.normal(0, sgm, size=size)
        X[1] = eps[1]
        for t in range(1, X.shape[0] - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * X[t - 1] * \
                    eps[t] / sgm**2 + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / 150)
            if dynamics == 'step':
                b[:2] = 0.1
                if (t < 300):
                    b[t + 1] = 0.1
                if (t >= 300):
                    b[t + 1] = 0.9
                if (t >= 600):
                    b[t + 1] = 0.1
            if dynamics == 'exp':
                b[t + 1] = np.exp(-(t + 1) / 500)
            X[t + 1] = a + b[t + 1] * X[t] + eps[t + 1]

    if model == 'poisson':
        X[0] = np.random.poisson(b[0])
        for t in range(size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * \
                    (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t+1) / 150)
            if dynamics == 'step':
                if t < 300:
                    b[t + 1] = 0.1
                if t >= 300:
                    b[t + 1] = 1
                if t >= 600:
                    b[t + 1] = 0.1
            if dynamics == 'exp':
                b[t + 1] = np.exp(-(t + 1) / 100)
            X[t + 1] = np.random.poisson(np.exp(b[t + 1]))
    return X, b


def model_loglikelihood(params, X, model):
    T = X.shape[0]
    b = np.zeros_like(X)
    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = params[0], params[1], params[2], params[3], params[4]
        for i in range(1, T - 1):
            b[i + 1] = omega + alpha * \
                (X[i] - a - b[i] * X[i - 1]) * X[i - 1] / sgm**2 + beta * b[i]

        sum = 0
        for i in range(T - 1):
            sum += (- 0.5 * np.log(sgm**2) - 0.5 *
                    (X[i + 1] - a - b[i + 1] * X[i])**2 / sgm**2)

    if model == 'poisson':
        alpha, beta, omega = params[0], params[1], params[2]
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        sum = 0
        for t in range(T):
            sum = sum + poisson.logpmf(X[t], mu=np.exp(b[t]))

    return - sum / T


def model_estimation(fun, X, init_params, model):
    res = minimize(fun, init_params, (X, model), method='BFGS',
                   options={'maxiter': 1000})
    std_err = np.sqrt([res.hess_inv[i, i] * (1 / X.shape[0])
                      for i in range(init_params.shape[0])])
    estimates = res.x
    estimates_up = res.x + std_err
    estimates_down = res.x - std_err

    T = X.shape[0]
    b = np.zeros_like(X)
    b_up = np.zeros_like(X)
    b_down = np.zeros_like(X)

    if model == 'autoregressive':
        for t in range(1, T - 1):
            b[t + 1] = estimates[1] + estimates[2] * X[t - 1] * \
                (X[t] - estimates[0] - b[t] * X[t - 1]) / \
                estimates[4]**2 + estimates[3] * b[t]

            b_up[t + 1] = estimates_up[1] + estimates_up[2] * X[t - 1] * \
                (X[t] - estimates_up[0] - b_up[t] * X[t - 1]) / \
                estimates_up[4]**2 + estimates_up[3] * b_up[t]

            b_down[t + 1] = estimates_down[1] + estimates_down[2] * X[t - 1] * \
                (X[t] - estimates_down[0] - b_down[t] * X[t - 1]) / \
                estimates_down[4]**2 + estimates_down[3] * b_down[t]

    if model == 'poisson':
        for t in range(T - 1):
            b[t + 1] = estimates[2] + estimates[0] * \
                (X[t] - np.exp(b[t])) * \
                (np.exp(b[t])) + estimates[1] * b[t]

            b_up[t + 1] = estimates_up[2] + estimates_up[0] * \
                (X[t] - np.exp(b_up[t])) * \
                (np.exp(b_up[t])) + estimates_up[1] * b_up[t]

            b_down[t + 1] = estimates_down[2] + estimates_down[0] * (X[t] - np.exp(
                b_down[t])) * (np.exp(b_down[t])) + estimates_down[1] * b_down[t]

    return [b, b_up, b_down], res, std_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-c", "--convergence", action='store_true')
    parser.add_argument("-m", "--model", type=int, help='0 for AR(1), 1 for Poisson')
    parser.add_argument("-d", "--dynamics", type=int, help='Dynamics for b: 0 for GAS, 1 for sinusodial, 2 for step function, 3 for exponential decay')

    args = parser.parse_args()
    plt.style.use('seaborn')
    # np.random.seed(666)

    if args.model == 0: model = 'autoregressive'
    if args.model == 1: model = 'poisson'
    if args.dynamics == 0: dynamics = 'gas'
    if args.dynamics == 1: dynamics = 'sin'
    if args.dynamics == 2: dynamics = 'step'
    if args.dynamics == 3: dynamics = 'exp'

    n = 1000
    if model == 'autoregressive':
        num_par = 5
        omega = 0.05
        alpha = 0.08
        beta = 0.46
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

    init_params = np.random.uniform(0, 0.1, size=num_par)
    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
    axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
    axs[1].plot(b[:n], 'k', label='Real', linewidth=1)

    B, res, std_err = model_estimation(
        model_loglikelihood, X, init_params, model)

    # b_test = np.zeros_like(b)
    # for t in range(1, b.shape[0] - 1):
    #     b_test[t + 1] = b_test[t] + res.x[2]/res.x[4]**2 * X[t-1] * (X[t] - res.x[0] - b_test[t] * X[t-1])

    if model == 'autoregressive':
        print('True values: ', [a, omega, alpha, beta, sgm])
    if model == 'poisson':
        print('True values: ', [omega, alpha, beta])
    print('Estimated values: ', res.x)
    print('Standard errors: ', std_err)

    axs[1].plot(B[0][:n], 'crimson', label='Filtered', linewidth=1)
    # axs[1].plot(b_test[:n], 'blue', label='Manual', linewidth=1)
    axs[1].fill_between(list(range(n)), B[2][:n], B[1]
                        [:n], color='crimson', alpha=0.3)

    axs[0].legend()
    axs[1].legend()
    plt.show()

    if args.convergence:
        N = 4
        est_par = np.empty(shape=(N, num_par))
        par = np.random.uniform(0, 1, size=(N, num_par))
        fun_val = []
        b_val = []
        for i in tqdm(range(len(par))):
            B, res, std_err = model_estimation(
                model_loglikelihood, X, init_params, model)
            fun_val.append(res.fun)
            est_par[i] = res.x
            b_val.append(B[0][-1])
        plt.figure(figsize=(12, 8), tight_layout=True)
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'k', linewidth=1)
        ax0.title.set_text('b(60) values')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'k', linewidth=1)
        ax1.title.set_text('Likelihood Evaluations')
        ax2 = plt.subplot(num_par, 3, 2)
        ax2.plot(par[:, 0], linewidth=1)
        ax2.title.set_text('Initial Omega')
        ax3 = plt.subplot(num_par, 3, 5)
        ax3.plot(par[:, 1], linewidth=1)
        ax3.title.set_text('Initial a')
        ax4 = plt.subplot(num_par, 3, 8)
        ax4.plot(par[:, 2], linewidth=1)
        ax4.title.set_text('Initial alpha')
        ax5 = plt.subplot(num_par, 3, 11)
        ax5.plot(par[:, 3], linewidth=1)
        ax5.title.set_text('Initial beta')
        ax11 = plt.subplot(num_par, 3, 14)
        ax11.plot(par[:, 4], linewidth=1)
        ax11.title.set_text('Initial sigma')

        ax6 = plt.subplot(num_par, 3, 3)
        ax6.plot(est_par[:, 0], 'k', linewidth=1)
        ax6.title.set_text('Estimated Omega')
        ax7 = plt.subplot(num_par, 3, 6)
        ax7.plot(est_par[:, 1], 'k', linewidth=1)
        ax7.title.set_text('Estimated a')
        ax8 = plt.subplot(num_par, 3, 9)
        ax8.plot(est_par[:, 2], 'k', linewidth=1)
        ax8.title.set_text('Estimated alpha')
        ax9 = plt.subplot(num_par, 3, 12)
        ax9.plot(est_par[:, 3], 'k', linewidth=1)
        ax9.title.set_text('Estimated beta')
        ax10 = plt.subplot(num_par, 3, 15)
        ax10.plot(est_par[:, 4], 'k', linewidth=1)
        ax10.title.set_text('Estimated sigma')
        plt.grid(True)
        plt.show()
