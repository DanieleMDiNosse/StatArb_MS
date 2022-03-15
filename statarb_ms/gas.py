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
import loglikelihood
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def likelihood_jac(params, X, b, model='autoregressive'):
    a, omega, alpha, beta, sgm = params
    num_par = params.shape[0]
    T = X.shape[0]
    XX = X[1:-1]
    Xm1 = X[:-2]
    Xp1 = X[2:]
    b = b[1:-1]

    dda =  (-(2*XX*Xm1*alpha/sgm**2 - 2)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/(2*sgm**2)).sum()
    ddw =  (XX*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**2).sum()
    ddal =  ((XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**4).sum()
    ddb = (XX*b*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**2).sum()

    if num_par == 5:
        dds =  (-2*alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**5 - 1.0/sgm + (-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)**2/sgm**3).sum()
        jac = - np.array([dda, ddw, ddal, ddb, dds])
    if num_par == 4:
        jac = - np.array([dda, ddw, ddal, ddb])

    return jac

def likelihood_hess(params, X, b, model='autoregressive'):
    num_par = params.shape[0]
    T = X.shape[0]
    omega, a, alpha, beta, sgm = params
    XX = X[1:-1]
    Xm1 = X[:-2]
    Xp1 = X[2:]
    b = b[1:-1]

    d2da2 =  (-(XX*Xm1*alpha/sgm**2 - 1)*(2*XX*Xm1*alpha/sgm**2 - 2)/(2*sgm**2)).sum()
    d2dw2 =  (-XX**2/sgm**2).sum()
    d2dal2 =  (- (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a)**2 / sgm**6).sum()
    d2db2 =  (-XX**2*b**2/sgm**2).sum()
    ddadw =  (XX*(2*XX*Xm1*alpha/sgm**2 - 2)/(2*sgm**2)).sum()
    ddadal =  (-XX*Xm1*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**4 + (2*XX*Xm1*alpha/sgm**2 - 2)*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/(2*sgm**4)).sum()
    ddadb =  (XX*b*(2*XX*Xm1*alpha/sgm**2 - 2)/(2*sgm**2)).sum()
    ddwdal =  (-XX*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**4).sum()
    ddwdb =  (-XX**2*b/sgm**2).sum()
    ddaldb =  (-XX*b*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**4).sum()

    if num_par == 5:
        ddsdw =  (2*XX*alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**5 - 2*XX*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**3).sum()
        ddsdal =  (2*alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)**2/sgm**7 - 4*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**5).sum()
        ddsdb =  (2*XX*alpha*b*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**5 - 2*XX*b*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**3).sum()
        ddsda =  (2*XX*Xm1*alpha*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**5 - alpha*(2*XX*Xm1*alpha/sgm**2 - 2)*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**5 + (2*XX*Xm1*alpha/sgm**2 - 2)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**3).sum()
        d2ds2 =  (-4*alpha**2*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)**2/sgm**8 + 14*alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)/sgm**6 + 1.0/sgm**2 - 3*(-XX*b*beta - XX*omega + Xp1 - a - alpha*(XX**2*Xm1 - XX*Xm1**2*b - XX*Xm1*a)/sgm**2)**2/sgm**4).sum()

        hess = - np.array([[d2da2, ddadw, ddadal, ddadb, ddsda], [ddadw, d2dw2, ddwdal, ddwdb, ddsdw], [ddadal, ddwdal,
                                   d2dal2, ddaldb, ddsdal], [ddadb, ddwdb, ddaldb, d2db2, ddsdb], [ddsda, ddsdw, ddsdal, ddsdb, d2ds2]])
    if num_par == 4:
        hess = - np.array([[d2da2, ddadw, ddadal, ddadb], [ddadw, d2dw2, ddwdal, ddwdb], [ddadal, ddwdal,
                                   d2dal2, ddaldb], [ddadb, ddwdb, ddaldb, d2db2]])

    return hess


def ML_errors(jac, hess_inv, params, X, specification):
    T = X.shape[0]
    num_par = params.shape[0]
    J = np.outer(jac, jac)

    if specification == 'mis':
        var = np.dot(np.dot(hess_inv, J), hess_inv)
        # ATTENTION: T is used here since you deleted it from likelihood
        std_err = np.sqrt([var[i, i] / T for i in range(num_par)])
    if specification == 'well':
        # ATTENTION: T is used here since you deleted it from likelihood
        std_err = np.sqrt([hess_inv[i, i] / T for i in range(num_par)])

    return std_err


def b_error(jac, hess_inv, deltas, X, specification):
    T = X.shape[0]
    num_par = deltas.shape[0]
    J = np.outer(jac, jac)
    var_par = np.dot(np.dot(hess_inv, J), hess_inv)
    print(deltas)

    s = 0
    for i in range(num_par - 1):
        for j in range(i + 1, num_par):
            if specification == 'well': A = hess_inv
            if specification == 'mis': A = var_par
            s += deltas[i] * A[i, j] * deltas[j]
        var = 1 / T * sum([A[i, i] * deltas[i] **
                    2 for i in range(num_par)]) + 2 / T * s
    std = np.sqrt(var)
    return std


def estimation(fun, X, init_params, method='L-BFGS-B', targeting_estimation=False, verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.zeros(shape=T)
    xi = np.zeros(shape=T)
    dbda, dbdw, dbdal, dbdb = np.zeros_like(X), np.zeros_like(
        X), np.zeros_like(X), np.zeros_like(X)

    if targeting_estimation:
        init_params0 = np.random.uniform(0, 1, size=2)
        sigma = 1
        res = minimize(loglikelihood.targeting_loglikelihood,
                       init_params0, X, method=method)
        omega, a = res.x
        b_bar = omega
        b[:2] = b_bar
        res = minimize(fun, init_params, (X, b_bar),
                       method=method, options={'eps': 1e-3})
        a, alpha, beta = res.x[0], res.x[1], res.x[2]
        omega = beta * (1 - b_bar)
        res = [omega, a, alpha, beta]

    else:
        res = minimize(fun, init_params, X, method=method,
                       options={'eps': 1e-3})
        omega, a, alpha, beta = res.x
        sgm = 1

    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(2.5)

    for t in range(1, T - 1):
        b[t + 1] = omega + alpha * xi[t] * X[t - 1] / sgm**2 + beta * b[t]
        xi[t + 1] = (X[t + 1] - a - b[t + 1] * X[t])

        dbda[t + 1] = - alpha * X[t - 1] / sgm**2 * (1 + X[t - 1] * dbda[t])
        dbdw[t + 1] = 1 - alpha / sgm**2 * X[t - 1]**2 * dbdw[t]
        dbdal[t + 1] = X[t - 1] / sgm**2 * \
            (X[t] - a - b[t] * X[t - 1] - alpha * X[t - 1] * dbdal[t])
        dbdb[t + 1] = X[t] - alpha / sgm**2 * X[t - 1]**2 * dbdb[t]

    delta_a = beta * dbda[-1] - alpha / sgm**2 * X[-2]
    delta_w = beta * dbdw[-1] + 1
    delta_al = beta * dbdal[-1] + 1 / sgm**2 * \
        (X[-1] - a - b[-1] * X[-2]) * X[-2]
    delta_b = beta * dbdb[-1] + b[-1]

    deltas = np.array([delta_a, delta_w, delta_al, delta_b])

    std = b_error(likelihood_hess(res.x, X, b), deltas, X)
    par_std = ML_errors(res.x, X, b, specification='mis')
    print('Std Errors: ',par_std)

    if visualization:
        plt.figure(figsize=(12, 5), tight_layout=True)
        plt.plot(b, linewidth=1, label='Filtered Data')
        plt.fill_between(list(range(T)), b + std, b - std,
                         color='crimson', label=r'$\sigma_b: {:.2f}$'.format(std), alpha=0.3)
        plt.legend()
        plt.grid(True)
        plt.title(r'$[\omega, a, \alpha, \beta]: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}$'.format(
            res.x[0], par_std[0], res.x[1], par_std[1], res.x[2], par_std[2], res.x[3], par_std[3]))
        plt.show()

    return b, a, xi, res, std


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

    if args.convergence:
        n_params = 4
        N = 100
        NN = 1000
        X = np.load(go_up(1) + '/saved_data/dis_res70.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        s_sgm, a_sgm, omega_sgm, beta_sgm, alpha_sgm = [], [], [], [], []
        day = np.random.randint(0, days, size=NN)
        stock = np.random.randint(0, n_stocks, size=NN)
        for i in tqdm(range(NN), desc='Covergence'):
            x = X[day[i], stock[i], :]
            a_val, omega_val, beta_val, alpha_val, s_val = np.empty(shape=N), np.empty(
                shape=N), np.empty(shape=N), np.empty(shape=N), np.empty(shape=N)
            par = np.random.uniform(0, 0.5, size=(N, 4))
            for i in range(N):
                b, a, xi, res, std = estimation(
                    reduced_loglikelihood.loglikelihood, x, par[i], targeting_estimation=args.targ_est, verbose=args.verbose, visualization=args.visualization)
                omega_val[i] = res.x[0]
                a_val[i] = res.x[1]
                alpha_val[i] = res.x[2]
                beta_val[i] = res.x[3]
                b = b[-1]
                m = a / (1 - b)
                sgm_eq = np.std(xi) * np.sqrt(1 / (1 - b * b))
                s_val[i] = -m / sgm_eq
            scaler = MinMaxScaler()
            a_val, omega_val, beta_val, alpha_val, s_val = scaler.fit_transform(a_val.reshape(-1, 1)), scaler.fit_transform(
                omega_val.reshape(-1, 1)), scaler.fit_transform(beta_val.reshape(-1, 1)), scaler.fit_transform(alpha_val.reshape(-1, 1)), scaler.fit_transform(s_val.reshape(-1, 1))
            s_sgm.append(s_val.std())
            alpha_sgm.append(alpha_val.std())
            omega_sgm.append(omega_val.std())
            beta_sgm.append(beta_val.std())
            a_sgm.append(a_val.std())
        plt.figure(figsize=(17, 4), tight_layout=True)
        ax0 = plt.subplot(1, 5, 1)
        ax0.hist(s_sgm, bins=40, color='blue', edgecolor='darkblue', alpha=0.6)
        ax0.title.set_text('Score')
        ax1 = plt.subplot(1, 5, 2)
        ax1.hist(a_sgm, bins=40, color='blue', edgecolor='darkblue', alpha=0.6)
        ax1.title.set_text(r'$\sigma_a$')
        ax2 = plt.subplot(1, 5, 3)
        ax2.hist(omega_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax2.title.set_text(r'$\sigma_{\omega}$')
        ax3 = plt.subplot(1, 5, 4)
        ax3.hist(alpha_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax3.title.set_text(r'$\sigma_{\alpha}$')
        ax4 = plt.subplot(1, 5, 5)
        ax4.hist(beta_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax4.title.set_text(r'$\sigma_{\beta}$')
        plt.savefig(go_up(1) + 'score_1000res_100init_60days.png')

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
        plt.show()

    else:
        X = np.load(
            '/run/media/danielemdn/0A9A624E5CE1F5FA/saved_data/A&L/dis_res.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        day = np.random.randint(days)
        stock = np.random.randint(n_stocks)
        print(day, stock)
        x = X[day, stock, :]
        if args.targ_est:
            init_params = np.random.uniform(0, 0.5, size=3)
            b, a, xi, res, std = estimation(loglikelihood.loglikelihood_after_targ,
                                            x, init_params, targeting_estimation=args.targ_est, visualization=True)
        else:
            init_params = np.random.uniform(0, 0.5, size=4)
            b, a, xi, res, std = estimation(loglikelihood.loglikelihood, x, init_params,
                                            targeting_estimation=args.targ_est, verbose=args.verbose, visualization=True)
        if (math.isnan(b[-1])) or (b[-1] < 0):
            print(b[-1])
