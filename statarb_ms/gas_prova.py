from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from tqdm import tqdm
import argparse
import time


def synt_data(a, omega, alpha, beta, sgm, model, dynamics, size):
    eps = np.random.normal(0, sgm, size=size)

    if model == 'autoregressive':
        X = np.zeros(shape=size)
        b = np.zeros(shape=size)
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
        X = np.zeros(shape=size)
        b = np.zeros(shape=size)
        X[0] = np.random.poisson(b[0])
        for t in range(size - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * \
                    (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * t / 200)
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

    return X, b, eps


def model_loglikelihood(params, X, model, target_est=False):
    if model == 'autoregressive':
        T = X.shape[0]
        b = np.zeros_like(X)
        if target_est:
            a, omega, sgm = params[0], params[1], params[2]
            alpha, beta = 0, 0
        else:
            a, alpha, omega, sgm = params[0], params[1], params[2], params[3]
            b_bar = 0.1155
            # b_bar = 0.14892448888413884
            # b_bar = 0.9451372490563206
            beta = 1 - omega / b_bar
            # omega = b_bar * (1 - beta)
            b[1] = b_bar

        for t in range(1, T - 1):
            b[t + 1] = omega + alpha * X[t - 1] * \
                (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]

        sum = 0
        for t in range(T - 1):
            sum += (- 0.5 * np.log(sgm**2) - 0.5 *
                    (X[t + 1] - a - b[t + 1] * X[t])**2 / sgm**2)
    if model == 'poisson':
        T = X.shape[0]
        alpha, beta, omega = params[0], params[1], params[2]
        b = np.zeros_like(X)
        for t in range(T - 1):
            b[t + 1] = omega + alpha * \
                (X[t] - np.exp(b[t])) * (np.exp(b[t])) + beta * b[t]

        sum = 0
        for t in range(T):
            sum = sum + poisson.logpmf(X[t], mu=np.exp(b[t]))
    return - sum / T


def model_estimation(fun, X, init_params, eps, model, long_term_mean=False):
    res = minimize(fun, init_params, args=(X, model, long_term_mean), method='BFGS',
                   options={'maxiter': 1000})
    std_err = np.sqrt([res.hess_inv[i, i] * (1 / X.shape[0])
                      for i in range(init_params.shape[0])])
    if model == 'autoregressive':
        X, b = np.zeros_like(X), np.zeros_like(X)
        X_up, b_up = np.zeros_like(X), np.zeros_like(X)
        X_down, b_down = np.zeros_like(X), np.zeros_like(X)

        if long_term_mean:
            alpha, beta = 0, 0
            a, omega, sgm = res.x[0], res.x[1], res.x[2]

            for t in range(1, X.shape[0] - 1):
                b[t + 1] = omega + alpha * X[t - 1] * \
                    (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
                X[t + 1] = a + b[t + 1] * X[t] + eps[t + 1]
        else:
            a, alpha, omega, sgm = res.x[0], res.x[1], res.x[2], res.x[3]
            b_bar = 0.1155
            # b_bar = 0.14892448888413884 # sin
            # b_bar = 0.9451372490563206 # step
            beta = 1 - omega / b_bar
            # omega = b_bar * (1 - beta)
            b[1] = b_bar
            b_up[1] = b_bar
            b_down[1] = b_bar
            a_up, alpha_up, beta_up, sgm_up = res.x[0] + std_err[0], res.x[1] + \
                std_err[1], res.x[2] + std_err[2], res.x[3] + \
                std_err[3]
            omega_up = b_bar * (1 - beta_up)
            a_down, alpha_down, beta_down, sgm_down = res.x[0] - std_err[0], res.x[1] - \
                std_err[1], res.x[2] - std_err[2], res.x[3] - \
                std_err[3]
            omega_down = b_bar * (1 - beta_down)

            for t in range(1, X.shape[0] - 1):
                b[t + 1] = omega + alpha * X[t - 1] * \
                    (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]

                b_up[t + 1] = omega_up + alpha_up * X[t - 1] * \
                    (X[t] - a_up - b_up[t] * X[t - 1]) / sgm_up**2 + beta_up * b_up[t]

                b_down[t + 1] = omega_down + alpha_down * X[t - 1] * \
                    (X[t] - a_down - b_down[t] * X[t - 1]) / sgm_down**2 + beta_down * b_down[t]

        return [X, X_up, X_down, b, b_up, b_down], res, std_err

    if model == 'poisson':
        f_est = np.zeros_like(X)
        f_est_up = np.zeros_like(X)
        f_est_down = np.zeros_like(X)
        par_est = res.x
        par_est_up = res.x + std_err
        par_est_down = res.x - std_err

        for t in range(f_est.shape[0] - 1):
            f_est[t + 1] = par_est[2] + par_est[0] * \
                (X[t] - np.exp(f_est[t])) * \
                (np.exp(f_est[t])) + par_est[1] * f_est[t]
            f_est_up[t + 1] = par_est_up[2] + par_est_up[0] * \
                (X[t] - np.exp(f_est_up[t])) * \
                (np.exp(f_est_up[t])) + par_est_up[1] * f_est_up[t]
            f_est_down[t + 1] = par_est_down[2] + par_est_down[0] * (X[t] - np.exp(
                f_est_down[t])) * (np.exp(f_est_down[t])) + par_est_down[1] * f_est_down[t]

        return [f_est, f_est_up, f_est_down], res, std_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-u", "--update", type=int, default=0,
                        help="Update scheme for gas estimation")
    parser.add_argument("-n", "--n_params", type=int, default=4,
                        help="Number of parameters to be estimates")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-vv", "--visualization", action='store_true')
    parser.add_argument("-c", "--convergence", action='store_true')
    parser.add_argument("-m", "--model", type=int)
    parser.add_argument("-t", "--target_est", action='store_true')

    args = parser.parse_args()
    plt.style.use('seaborn')
    np.random.seed(666)

    if args.model == 0:
        model = 'autoregressive'
    if args.model == 1:
        model = 'poisson'

    n = 1000
    if model == 'poisson':
        alpha = 0.081
        beta = -0.395
        omega = 0.183
        a, sgm = 0, 0
    if model == 'autoregressive':
        omega = 0.05
        alpha = 0.08
        beta = 0.06
        sgm = 0.1
        a = 0.1

    if (args.target_est) or (model == 'poisson'):
        init_params = np.random.uniform(0, 0.1, size=3)
    else:
        init_params = np.random.uniform(0, 0.1, size=4)

    X, b, eps = synt_data(a, omega, alpha, beta, sgm, model,
                          dynamics='gas', size=n)

    rec, res, std_err = model_estimation(
        model_loglikelihood, X, init_params, eps, model, args.target_est)
    if args.target_est:
        print(f'Long term mean for b(t): {rec[3][-1]}')

    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
    if model == 'autoregressive':
        print('True values: ',
              [a, omega, alpha, sgm])
        axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
        axs[0].set_ylabel('X')
        axs[0].plot(rec[0][:n], 'green', label='Filtered', linewidth=1)
        axs[0].fill_between(list(range(n)), rec[2][:n], rec[1]
                            [:n], color='green', alpha=0.3)
        axs[1].plot(b[:n], 'k', label='Real', linewidth=1)
        axs[1].set_ylabel('b')
        axs[1].plot(rec[3][:n], 'crimson', label='Filtered', linewidth=1)
        axs[1].fill_between(list(range(n)), rec[5][:n], rec[4]
                            [:n], color='crimson', alpha=0.3)
    if model == 'poisson':
        print('True values: ',
              [a, omega, alpha])
        axs[0].plot(X[:n], linewidth=1, label='Real')
        axs[0].set_ylabel('X')
        axs[1].plot(b[:n], 'k', linewidth=1, label='Real')
        axs[1].set_ylabel('f')
        axs[1].plot(rec[0][:n], 'crimson', label='Filtered', linewidth=1)
        axs[1].fill_between(list(range(n)), rec[1][:n], rec[2]
                            [:n], color='crimson', alpha=0.3)
    print('Estimated values: ', res.x)
    print('Standard errors: ', std_err)

    axs[0].legend()
    axs[1].legend()
    plt.show()

    if args.convergence:
        N = 100
        est_par = np.empty(shape=(N, 4))
        stderr_par = np.empty(shape=(N, 4))
        par = np.random.uniform(0, 1, size=(N, 4))
        fun_val = []
        b_val = []
        for i in tqdm(range(len(par))):
            rec, res, std_err = model_estimation(
                model_loglikelihood, X, init_params, eps, model, args.target_est)
            fun_val.append(res.fun)
            est_par[i] = res.x
            stderr_par[i] = std_err
            b_val.append(rec[3][-1])
        plt.figure(figsize=(12, 8), tight_layout=True)
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'k', linewidth=1)
        ax0.title.set_text('b(60) values')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'k', linewidth=1)
        ax1.title.set_text('Likelihood Evaluations')
        ax2 = plt.subplot(args.n_params, 3, 2)
        ax2.plot(par[:, 0], linewidth=1)
        ax2.title.set_text('Initial a')
        ax3 = plt.subplot(args.n_params, 3, 5)
        ax3.plot(par[:, 1], linewidth=1)
        ax3.title.set_text('Initial omega')
        ax4 = plt.subplot(args.n_params, 3, 8)
        ax4.plot(par[:, 2], linewidth=1)
        ax4.title.set_text('Initial alpha')
        ax5 = plt.subplot(args.n_params, 3, 11)
        ax5.plot(par[:, 3], linewidth=1)
        ax5.title.set_text('Initial sigma')

        ax6 = plt.subplot(args.n_params, 3, 3)
        ax6.plot(est_par[:, 0], 'k', linewidth=1)
        ax6.errorbar(np.arange(0,N), est_par[:, 0], yerr=stderr_par[:,0], elinewidth=0.5, capsize=3, capthick=1, errorevery=int(N/4))
        ax6.hlines(a, 0, N, linestyles='dashed', linewidth=1)
        ax6.title.set_text('Estimated a')
        ax7 = plt.subplot(args.n_params, 3, 6)
        ax7.plot(est_par[:, 1], 'k', linewidth=1)
        ax7.errorbar(np.arange(0,N), est_par[:, 1], yerr=stderr_par[:,1], elinewidth=0.5, capsize=3, capthick=1, errorevery=int(N/4))
        ax7.hlines(omega, 0, N, linestyles='dashed', linewidth=1)
        ax7.title.set_text('Estimated omega')
        ax8 = plt.subplot(args.n_params, 3, 9)
        ax8.plot(est_par[:, 2], 'k', linewidth=1)
        ax8.errorbar(np.arange(0,N), est_par[:, 2], yerr=stderr_par[:,2], elinewidth=0.5, capsize=3, capthick=1, errorevery=int(N/4))
        ax8.hlines(alpha, 0, N, linestyles='dashed', linewidth=1)
        ax8.title.set_text('Estimated alpha')
        ax9 = plt.subplot(args.n_params, 3, 12)
        ax9.plot(est_par[:, 3], 'k', linewidth=1)
        ax9.errorbar(np.arange(0,N), est_par[:, 3], yerr=stderr_par[:,3], elinewidth=0.5, capsize=3, capthick=1, errorevery=int(N/4))
        ax9.hlines(sgm, 0, N, linestyles='dashed', linewidth=1)
        ax9.title.set_text('Estimated sigma')
        plt.grid(True)
        plt.show()
