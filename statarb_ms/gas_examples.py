from scipy.optimize import minimize
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import time
from regression_parameters import regression
from gas import ML_errors, likelihood_jac, likelihood_hess


def synt_data(model, *args, dynamics, size):
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)

    if model == 'autoregressive':
        eps = np.random.normal(0, sgm, size=size)
        X[1] = eps[1]
        for t in range(1, X.shape[0] - 1):
            if dynamics == 'gas':
                b[t + 1] = omega + alpha * X[t - 1] * (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]
            if dynamics == 'sin':
                b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / 150)
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
                b[t + 1] = w + k * b[t] + np.random.normal(0,0.1)
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
        a, omega, alpha, beta, sgm = params[0], params[1], params[2], params[3], params[4]
        # omega = 0.180 * (1 - beta) # sin
        # omega = 0.16339 * (1 - beta) #gas
        # omega = 0.3192 * (1 - beta) #step
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

    return - sum


def model_estimation(fun, X, b, init_params, model):
    res = minimize(fun, init_params, (X, model), method='BFGS',
                   options={'maxiter': 1000})
    # std_err = np.sqrt([res.hess_inv[i, i] for i in range(init_params.shape[0])])
    std_err = ML_errors(res.x, X, b)
    print('hess_inv by scipy \n', res.hess_inv)
    estimates = res.x
    estimates_up = res.x + std_err
    estimates_down = res.x - std_err

    T = X.shape[0]
    b = np.zeros_like(X)
    dbda, dbdw, dbdal, dbdb, dbds = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

    if model == 'autoregressive':
        a, omega, alpha, beta, sgm = estimates[0], estimates[1], estimates[2], estimates[3], estimates[4]
        # estimates[1] = 0.180 * (1 - estimates[3]) #sin
        # estimates[1] = 0.16339 * (1 - estimates[3]) #gas
        # estimates[1] = 0.3192 * (1 - estimates[3]) # step
        for t in range(1, T - 1):
            b[t + 1] = omega + alpha * X[t - 1] * (X[t] - a - b[t] * X[t - 1]) / sgm**2 + beta * b[t]

            dbda[t + 1] = - alpha*X[t - 1]/sgm**2 * (1 + X[t - 1]*dbda[t])
            dbdw[t + 1] = 1 - alpha/sgm**2 * X[t - 1]**2*dbdw[t]
            dbdal[t + 1] = X[t - 1]/sgm**2 * (X[t] - a - b[t]*X[t - 1] - alpha*X[t - 1]*dbdal[t])
            dbdb[t + 1] = X[t] - alpha/sgm**2 * X[t - 1]**2*dbdb[t]
            dbds[t + 1] = alpha*X[t - 1]/sgm**2 * (2/(sgm) * (b[t]*X[t - 1] - X[t] + a) - X[t - 1]*dbds[t])

        delta_a = beta * dbda[-1] - alpha/sgm**2 * X[-2]
        delta_w = beta * dbdw[-1] + 1
        delta_al = beta * dbdal[-1] + 1/sgm**2 * (X[-1] - a - b[-1]*X[-2]) * X[-2]
        delta_b = beta*dbdb[-1] + b[-1]
        delta_s = beta * dbds[-1] - 2*alpha/(sgm**3) * (X[-1] - a - b[-1]*X[-2])*X[-2]

        deltas = np.array([delta_a, delta_w, delta_al, delta_b, delta_s])
    if model == 'poisson':
        for t in range(T - 1):
            b[t + 1] = estimates[2] + estimates[0] * (X[t] - np.exp(b[t])) * (np.exp(b[t])) + estimates[1] * b[t]

    return b, res, std_err, deltas

def b_error(hess, deltas, X):
    T = X.shape[0]
    Q = deltas.shape[0]
    hess_inv = np.linalg.inv(hess)
    s = 0
    for i in range(Q-1):
        for j in range(i+1,Q):
            s += deltas[i]*hess_inv[i,j]*deltas[j]
    print('My hess_inv \n', np.linalg.inv(hess))
    print('HERE \n', hess_inv, np.array(deltas)**2)
    var = 1/T * sum([hess_inv[i,i] * deltas[i]**2 for i in range(Q)]) + 2/T * s
    std = np.sqrt(var)
    return std

def LM_test_statistic(Xr, params):
    omega, a, sigma = params[0], params[1], params[2]
    T = Xr.shape[0]
    Y = np.zeros_like(Xr)
    X = np.zeros_like(Xr)
    for t in range(1, T):
        Y[t] = 1 / (Xr[t - 1] / sigma**2 * (Xr[t] - omega * Xr[t - 1] - a))
    for t in range(2, T):
        X[t] = Xr[t - 1] / sigma**2 * (Xr[t] - omega * Xr[t - 1] - a)
    c_w, c_a, conf_intervals, residuals, predictions, rsquared = regression(X, Y)
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
    parser.add_argument("-m", "--model", type=int, help='0 for AR(1), 1 for Poisson')
    parser.add_argument("-d", "--dynamics", type=int, help='Dynamics for b: 0 for GAS, 1 for sinusodial, 2 for step function, 3 for exponential decay')
    parser.add_argument("-lm", "--lmtest", action='store_true', help='Lm test')
    parser.add_argument("-mr", "--meanreversion", action='store_true', help='Mean reversion analysis')

    args = parser.parse_args()
    plt.style.use('seaborn')
    np.random.seed(666)

    if args.model == 0: model = 'autoregressive'
    if args.model == 1: model = 'poisson'
    if args.dynamics == 0: dynamics = 'gas'
    if args.dynamics == 1: dynamics = 'sin'
    if args.dynamics == 2: dynamics = 'step'
    if args.dynamics == 3: dynamics = 'ar1'

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
        params = [omega, a , sgm]
        ESS = LM_test_statistic(X, params)
        print(ESS)
        time.sleep(10)

    if args.meanreversion:
        plt.figure(figsize=(12,5), tight_layout=True)
        ax0 = plt.subplot(2, 1, 1)
        ax0.plot(X, linewidth=1)
        ax1 = plt.subplot(2, 1, 2)
        ax1.plot(b, linewidth=1)
        ax0.grid(True)
        ax1.grid(True)
        plt.show()

    else:
        init_params = np.random.uniform(0, 0.5, size=num_par)
        fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(14, 5))
        axs[0].plot(X[:n], 'k', label='Real', linewidth=1)
        axs[1].plot(b[:n], 'k', label='Real', linewidth=1)
        axs[0].set_ylabel('X')
        axs[1].set_ylabel('b')

        B, res, std_err, deltas = model_estimation(
            model_loglikelihood, X, b, init_params, model)
        hess = likelihood_hess(res.x, X, B)
        std = b_error(hess, deltas, X)

        MSE = mean_squared_error(b, B)
        # print(f'MSE: {MSE}')
        ### Targeting estimation ###
        # b_bar = w / (1 - beta) -> (1 - beta) * b_bar = w

        if model == 'autoregressive':
            print('True values: ', [a, omega, alpha, beta, sgm])
        if model == 'poisson':
            print('True values: ', [omega, alpha, beta])
        print('Initial values: ', init_params)
        print('Estimated values: ', res.x)
        print('Standard errors: ', std_err)
        print('Error on b: ', std)

        axs[1].plot(B[:n], 'crimson', label='Filtered', linewidth=1)
        # axs[1].plot(b_test[:n], 'blue', label='Manual', linewidth=1)
        axs[1].fill_between(list(range(n)), B[:n]+3*std, B[:n]-3*std, color='crimson', alpha=0.3)

        axs[0].legend()
        axs[1].legend()
        plt.show()

    if args.convergence:
        N = 10
        est_par = np.empty(shape=(N, num_par))
        stderr_par = np.empty(shape=(N, num_par))
        par = np.random.uniform(0, 1, size=(N, num_par))
        fun_val = []
        b_val = []
        for i in tqdm(range(len(par))):
            B, res, std_err = model_estimation(
                model_loglikelihood, X, par[i], model)
            fun_val.append(res.fun)
            est_par[i] = res.x
            stderr_par[i] = std_err
            b_val.append(B[0][-1])
        plt.figure(figsize=(12, 8))
        ax0 = plt.subplot(2, 3, 4)
        ax0.plot(b_val, 'crimson', linewidth=1)
        ax0.title.set_text('b(60) values')
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(fun_val, 'crimson', linewidth=1)
        ax1.title.set_text('Likelihood Evaluations')
        ax2 = plt.subplot(num_par, 3, 2)
        ax2.plot(par[:, 0], 'slateblue', linewidth=1)
        ax2.title.set_text('Initial Omega')
        ax2.tick_params(labelbottom=False)
        ax3 = plt.subplot(num_par, 3, 5)
        ax3.plot(par[:, 1], 'slateblue', linewidth=1)
        ax3.title.set_text('Initial a')
        ax3.tick_params(labelbottom=False)
        ax4 = plt.subplot(num_par, 3, 8)
        ax4.plot(par[:, 2], 'slateblue', linewidth=1)
        ax4.title.set_text('Initial alpha')
        ax4.tick_params(labelbottom=False)
        ax5 = plt.subplot(num_par, 3, 11)
        ax5.plot(par[:, 3], 'slateblue', linewidth=1)
        ax5.title.set_text('Initial beta')
        ax5.tick_params(labelbottom=False)
        ax11 = plt.subplot(num_par, 3, 14)
        ax11.plot(par[:, 4], 'slateblue', linewidth=1)
        ax11.title.set_text('Initial sigma')

        ax6 = plt.subplot(num_par, 3, 3)
        ax6.plot(est_par[:, 0], 'g', linewidth=1)
        ax6.errorbar(np.arange(0,N), est_par[:, 0], yerr=stderr_par[:,0], elinewidth=0.6, capsize=3, capthick=1)
        ax6.hlines(omega, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax6.title.set_text('Estimated Omega')
        ax6.tick_params(labelbottom=False)
        ax7 = plt.subplot(num_par, 3, 6)
        ax7.plot(est_par[:, 1], 'green', linewidth=1)
        ax7.errorbar(np.arange(0,N), est_par[:, 1], yerr=stderr_par[:,1], elinewidth=0.6, capsize=3, capthick=1)
        ax7.hlines(a, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax7.title.set_text('Estimated a')
        ax7.tick_params(labelbottom=False)
        ax8 = plt.subplot(num_par, 3, 9)
        ax8.plot(est_par[:, 2], 'green', linewidth=1)
        ax8.errorbar(np.arange(0,N), est_par[:, 2], yerr=stderr_par[:,2], elinewidth=0.6, capsize=3, capthick=1)
        ax8.hlines(alpha, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax8.title.set_text('Estimated alpha')
        ax8.tick_params(labelbottom=False)
        ax9 = plt.subplot(num_par, 3, 12)
        ax9.plot(est_par[:, 3], 'green', linewidth=1)
        ax9.errorbar(np.arange(0,N), est_par[:, 3], yerr=stderr_par[:,3], elinewidth=0.6, capsize=3, capthick=1)
        ax9.hlines(beta, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax9.title.set_text('Estimated beta')
        ax9.tick_params(labelbottom=False)
        ax10 = plt.subplot(num_par, 3, 15)
        ax10.plot(est_par[:, 4], 'green', linewidth=1)
        ax10.errorbar(np.arange(0,N), est_par[:, 4], yerr=stderr_par[:,4], elinewidth=0.6, capsize=3, capthick=1)
        ax10.hlines(sgm, 0, N, 'darkgreen', linestyle='dashed', linewidth=1)
        ax10.title.set_text('Estimated sigma')
        plt.grid(True)
        plt.show()
