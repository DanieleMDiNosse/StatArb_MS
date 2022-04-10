import argparse
import math
import time

import loglikelihood
import matplotlib.pyplot as plt
import numpy as np
from makedir import go_up
from numba import njit, prange
from scipy import signal
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def likelihood_jac(params, X, b, model='autoregressive'):
    a, omega, alpha, beta, sgm = params
    num_par = params.shape[0]
    T = X.shape[0]
    XX = X[1:-1]
    Xm1 = X[:-2]
    Xp1 = X[2:]
    b = b[1:-1]

    dda = (-(2 * XX * Xm1 * alpha / sgm**2 - 2) * (-XX * b * beta - XX * omega + Xp1 - a -
           alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / (2 * sgm**2)).sum()
    ddw = (XX * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX **
           2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**2).sum()
    ddal = ((XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) * (-XX * b * beta - XX * omega +
            Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**4).sum()
    ddb = (XX * b * (-XX * b * beta - XX * omega + Xp1 - a - alpha *
           (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**2).sum()

    if num_par == 5:
        dds = (-2 * alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 *
               a) / sgm**2) / sgm**5 - 1.0 / sgm + (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2)**2 / sgm**3).sum()
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

    d2da2 = (-(XX * Xm1 * alpha / sgm**2 - 1) * (2 * XX *
             Xm1 * alpha / sgm**2 - 2) / (2 * sgm**2)).sum()
    d2dw2 = (-XX**2 / sgm**2).sum()
    d2dal2 = (- (XX**2 * Xm1 - XX * Xm1**2 *
              b - XX * Xm1 * a)**2 / sgm**6).sum()
    d2db2 = (-XX**2 * b**2 / sgm**2).sum()
    ddadw = (XX * (2 * XX * Xm1 * alpha / sgm**2 - 2) / (2 * sgm**2)).sum()
    ddadal = (-XX * Xm1 * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm **
              2) / sgm**4 + (2 * XX * Xm1 * alpha / sgm**2 - 2) * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / (2 * sgm**4)).sum()
    ddadb = (XX * b * (2 * XX * Xm1 * alpha / sgm**2 - 2) / (2 * sgm**2)).sum()
    ddwdal = (-XX * (XX**2 * Xm1 - XX * Xm1**2 *
              b - XX * Xm1 * a) / sgm**4).sum()
    ddwdb = (-XX**2 * b / sgm**2).sum()
    ddaldb = (-XX * b * (XX**2 * Xm1 - XX * Xm1 **
              2 * b - XX * Xm1 * a) / sgm**4).sum()

    if num_par == 5:
        ddsdw = (2 * XX * alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**5 - 2 * XX * (-XX * b * beta -
                 XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**3).sum()
        ddsdal = (2 * alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a)**2 / sgm**7 - 4 * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a)
                  * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**5).sum()
        ddsdb = (2 * XX * alpha * b * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**5 - 2 * XX * b * (-XX * b *
                 beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**3).sum()
        ddsda = (2 * XX * Xm1 * alpha * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**5 - alpha * (2 * XX * Xm1 * alpha / sgm**2 - 2) * (XX**2 * Xm1 -
                 XX * Xm1**2 * b - XX * Xm1 * a) / sgm**5 + (2 * XX * Xm1 * alpha / sgm**2 - 2) * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**3).sum()
        d2ds2 = (-4 * alpha**2 * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a)**2 / sgm**8 + 14 * alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 *
                 Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2) / sgm**6 + 1.0 / sgm**2 - 3 * (-XX * b * beta - XX * omega + Xp1 - a - alpha * (XX**2 * Xm1 - XX * Xm1**2 * b - XX * Xm1 * a) / sgm**2)**2 / sgm**4).sum()

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
        std_err = np.sqrt([var[i, i] / T for i in range(num_par)])
    if specification == 'correct':
        std_err = np.sqrt([hess_inv[i, i] / T for i in range(num_par)])

    return std_err


def b_error(jac, hess_inv, deltas, X, specification):
    T = X.shape[0]
    num_par = hess_inv.shape[0]
    J = np.outer(jac, jac)
    var_par = np.dot(np.dot(hess_inv, J), hess_inv)

    s = 0
    for i in range(num_par - 1):
        for j in range(i + 1, num_par):
            if specification == 'correct':
                A = hess_inv
            if specification == 'mis':
                A = var_par
            s += deltas[i] * A[i, j] * deltas[j]
    var = 1 / T * sum([A[i, i] * deltas[i] **
                       2 for i in range(num_par)]) + 2 / T * s
    std = np.sqrt(var)
    return std


def estimation(X, method='L-BFGS-B', targeting_estimation=False, verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.zeros(shape=T)
    xi = np.zeros(shape=T)
    res_iter = np.zeros(shape=(2,5))

    if targeting_estimation:
        init_params0 = np.random.uniform(0, 1, size=2)
        sgm = 1
        res = minimize(loglikelihood.targeting_loglikelihood,
                       init_params0, X, method=method, options={'eps': 1e-3})

        while res.success == False:
            init_params0 = np.random.uniform(0, 1, size=2)
            res = minimize(loglikelihood.targeting_loglikelihood,
                           init_params0, X, method=method, options={'eps': 1e-3})

        omega, a = res.x
        b_bar = omega
        b[:2] = b_bar
        init_params = np.random.uniform(0, 1, size=3)
        res = minimize(loglikelihood.loglikelihood_after_targ, init_params, (X, b_bar),
                       method=method, options={'eps': 1e-3})

        while res.success == False:
            init_params = np.random.uniform(0, 1, size=3)
            res = minimize(loglikelihood.loglikelihood_after_targ,
                           init_params, X, method=method, options={'eps': 1e-3})

        a, alpha, beta = res.x[0], res.x[1], res.x[2]
        omega = beta * (1 - b_bar)
        res = [omega, a, alpha, beta]

    else:
        # init_params = np.random.uniform(0, 1, size=4)
        # res = minimize(loglikelihood.loglikelihood, init_params, X, method=method,
        #                options={'eps': 1e-3})
        #
        # while res.success == False:
        #     init_params = np.random.uniform(0, 1, size=4)
        #     res = minimize(loglikelihood.loglikelihood,
        #                    init_params, X, method=method, options={'eps': 1e-3})

        for i in range(res_iter.shape[0]):
            init_params = np.random.uniform(0, 1, size=4)
            res = minimize(loglikelihood.loglikelihood, init_params, X, method='Nelder-Mead')
            if np.isnan(res.fun) == False:
                res_iter[i, :-1] = res.x
                res_iter[i, -1] = res.fun

        init_params = res_iter[np.where(res_iter[:,4] == res_iter[:,4].min())][0][:4]


        # while res.success == False:
        #     init_params = np.random.uniform(0, 1, size=4)
        res = minimize(loglikelihood.loglikelihood,
                       init_params, X, method='Nelder-Mead')
        res = minimize(loglikelihood.loglikelihood,
                       res.x, X, method='BFGS', options={'maxiter': 1})

        omega, a, alpha, beta = res.x
        sgm = 1

    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(2.5)

    for t in range(1, T - 1):
        b[t + 1] = omega + alpha * xi[t] * X[t - 1] / sgm**2 + beta * b[t]
        xi[t + 1] = (X[t + 1] - a - b[t + 1] * X[t])

    if visualization:
        plt.figure(figsize=(12, 5), tight_layout=True)
        plt.plot(b, linewidth=1, label='Filtered Data')
        # plt.fill_between(list(range(T)), b + std, b - std,
        #                  color='crimson', label=r'$\sigma_b: {:.2f}$'.format(std), alpha=0.3)
        plt.legend()
        plt.grid(True)
        plt.title(r'$[\omega, a, \alpha, \beta]: {:.4f} , {:.4f} , {:.4f} , {:.4f} $'.format(
            res.x[0],  res.x[1],  res.x[2], res.x[3]))
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
    # np.random.seed(666)

    if args.update == 0:
        update = 'gaussian'
    if args.update == 1:
        update = 'logistic'

    if args.convergence:
        n_params = 4
        N = 1000
        NN = 1
        X = np.load(go_up(1) + '/saved_data/dis_res60.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        b_sgm = np.empty(shape=NN)
        day = np.random.randint(0, days, size=NN)
        day = 118
        stock = np.random.randint(0, n_stocks, size=NN)
        stock = 345
        for i in range(NN):
            print(f'Day: {day}, Stock: {stock}')
            x = X[day, stock, :]
            results = np.empty(shape=(N, n_params))
            for j in tqdm(range(N), desc='Covergence'):
                b, a, xi, res = estimation(
                    x, targeting_estimation=args.targ_est, verbose=args.verbose, visualization=args.visualization)
                results[j] = res.x
            # scaler = MinMaxScaler()
            # b_val = scaler.fit_transform(b_val.reshape(-1, 1))
            # b_sgm[i] = b_val.std()
        # np.save(go_up(1) + f'/saved_data/convergence_day{day}_stock{stock}_NM', results)
        omega, a, alpha, beta = results[:, 0], results[:, 1], results[:, 2], results[:, 3]

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
        plt.show()

        plt.figure(figsize=(12, 8), tight_layout=True)
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(omega, color='darkblue', alpha=0.6)
        ax1.title.set_text(r'$\hat{\omega}$')
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(a, color='darkblue', alpha=0.6)
        ax2.title.set_text(r'$\hat{a}$')
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(alpha, color='darkblue', alpha=0.6)
        ax3.title.set_text(r'$\hat{\alpha}$')
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(beta, color='darkblue', alpha=0.6)
        ax4.title.set_text(r'$\hat{\beta}$')
        plt.show()
        # plt.savefig(go_up(1) + 'b_1000res_100init_100days_targ.png')

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
        X = np.load(go_up(1) + '/saved_data/dis_res60.npy')
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
