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

def likelihood_jac(estimates, X, b):
    num_par = estimates.shape[0]
    if num_par == 4:
        omega, a, alpha, beta = estimates[0], estimates[1], estimates[2], estimates[3]
    if num_par == 5:
        omega, a, alpha, beta, sgm = estimates[0], estimates[1], estimates[2], estimates[3], estimates[4]

    stuff = (X[2:] - a - omega*X[1:-1] - alpha*(X[1:-1]**2*X[:-2] - a - X[:-2]*X[1:-1] - b[1:-1]*X[:-2]**2*X[1:-1]) - beta*b[1:-1]*X[1:-1])/sgm**2

    dda = - (stuff * (alpha/sgm**2*X[:-2]*X[1:-1] - 1)).sum()
    ddw = (X[1:-1] * stuff).sum()
    ddal = -(stuff * (X[1:-1]*X[:-2]/sgm**2 * (b[1:-1]*X[:-2] + a - X[1:-1]))).sum()
    ddb = (stuff*b[1:-1]*X[1:-1]).sum()
    if num_par == 5:
        dds = (-1/sgm**2 + 1/sgm**3*stuff - 2*alpha/sgm**5*stuff*(X[1:-1]**2*X[:-2] - a*X[1:-1]*X[:-2] - b[1:-1]*X[1:-1]*X[:-2]**2)).sum()
        jac = [dda, ddw, ddal, ddb, dds]
    else:
        jac = [dda, ddw, ddal, ddb]

    return np.array(jac)

def likelihood_hess(estimates, X, b):
    num_par = estimates.shape[0]
    if num_par == 4:
        omega, a, alpha, beta = estimates[0], estimates[1], estimates[2], estimates[3]
    if num_par == 5:
        omega, a, alpha, beta, sgm = estimates[0], estimates[1], estimates[2], estimates[3], estimates[4]

    stuff = (X[2:] - a - omega*X[1:-1] - alpha*(X[1:-1]**2*X[:-2] - a - X[:-2]*X[1:-1] - b[1:-1]*X[:-2]**2*X[1:-1]) - beta*b[1:-1]*X[1:-1])/sgm**2
    stuff1 = (X[1:-1]**2*X[:-2] - a*X[:-2]*X[1:-1] - b[1:-1]*X[:-2]**2*X[1:-1])

    d2da2 = - ((alpha * X[1:] * X[:-1] - 1)**2).sum()
    d2dw2 = - (X**2).sum()
    d2dal2 = - ((X[:-1] * X[1:] * (b[1:] * X[:-1] + a - X[1:]))**2).sum()
    d2db2 = - ((b * X)**2).sum()

    ddadw = (X[1:] * (alpha * X[1:] * X[:-1] - 1)).sum()
    ddadal = - (X[:-2] * X[1:-1] * (X[2:] - a - omega * X[1:-1] - alpha * (X[1:-1]**2 * X[:-2] - a * X[:-2] * X[1:-1] - b[1:-1] * X[:-2]**2 * X[1:-1]) - beta * b[1:-1] * X[1:-1]) + (alpha * X[:-2]* X[1:-1] - 1) * (X[:-2] * X[1:-1] * (b[1:-1] * X[:-2] + a - X[1:-1]))).sum()
    ddadb = ((alpha * X[1:] * X[:-1] - 1) * b[1:] * X[1:]).sum()

    ddwdal = (X[:-1] * X[1:]**2 * (b[1:] * X[:-1] + a - X[1:])).sum()
    ddwdb = - (b * X**2).sum()
    ddaldb = (X[1:]**2 * X[:-1] * b[1:] * (b[1:] * X[:-1] + a - X[1:])).sum()

    if num_par == 5:
        d2ds2 = (1/sgm**2 * (1 + 1/sgm**2*(3*stuff**2 + 2/sgm**2*(2*alpha*stuff*stuff1 + 5*stuff*stuff1 - 2*alpha**2*stuff1/sgm**4)))).sum()
        ddsdw = (- 2*stuff*X[1:-1]/sgm**3 + 2*alpha*X[1:-1]*stuff1/sgm**5).sum()
        ddsdal = (2*stuff1/sgm**5*(alpha*stuff1/sgm**2 - 2*stuff)).sum()
        ddsdb = (- 2*stuff*b[1:-1]*X[1:-1]/sgm**3 + 2*alpha*stuff1*b[1:-1]*X[1:-1]/sgm**5).sum()
        ddsda = (2*stuff/sgm**3 * (alpha/sgm**2*X[:-2]*X[1:-1] - 1) - 2*alpha/sgm**5*(alpha/sgm**2*X[:-2]*X[1:-1] - 1)*stuff1 + 2*alpha*stuff/sgm**5*X[:-2]*X[1:-1]).sum()
        hess = - np.array([[d2da2, ddadw, ddadal, ddadb, ddsda], [ddadw, d2dw2, ddwdal, ddwdb, ddsdw], [ddadal, ddwdal, d2dal2, ddaldb, ddsdal], [ddadb, ddwdb, ddaldb, d2db2, ddsdb], [ddsda, ddsdw, ddsdal, ddsdb, d2ds2]])
    else:
        hess = - np.array([[d2da2, ddadw, ddadal, ddadb], [ddadw, d2dw2, ddwdal, ddwdb], [ddadal, ddwdal, d2dal2, ddaldb], [ddadb, ddwdb, ddaldb, d2db2]])
    return np.array(hess)

def ML_errors(estimates, X, b):
    jac = likelihood_jac(estimates, X, b)
    hess = likelihood_hess(estimates, X, b)

    J = np.outer(jac,jac)
    hess_inv = np.linalg.inv(hess)

    var = np.dot(hess_inv, np.dot(J, hess_inv))
    std_err = np.sqrt([var[i,i]/X.shape[0] for i in range(estimates.shape[0])]) # ATTENTION: T is used here since you deleted it from likelihood

    return std_err

def estimation(fun, X, init_params, method='L-BFGS-B', targeting_estimation=False, verbose=False, visualization=False):
    '''Estimation of GAS parameters'''
    T = X.shape[0]
    b = np.zeros(shape=T)
    xi = np.zeros(shape=T)

    if targeting_estimation:
        init_params0 = np.random.uniform(0, 1, size=2)
        sigma = 1
        res = minimize(loglikelihood.targeting_loglikelihood, init_params0, X, method=method)
        omega, a = res.x[0], res.x[1]
        b_bar = omega
        b[:2] = b_bar
        res = minimize(fun, init_params, (X, b_bar), method=method, options={'eps': 1e-3})
        a, alpha, beta = res.x[0], res.x[1], res.x[2]
        omega = beta * (1 - b_bar)
        res = [omega, a, alpha, beta]

    else:
        res = minimize(fun, init_params, X, method=method, options={'eps': 1e-3})
        omega, a, alpha, beta = res.x[0], res.x[1], res.x[2], res.x[3]
        sigma = 1

    if verbose:
        print(f'Initial guess: \n {init_params}')
        print(res)
        time.sleep(2.5)

    for i in range(1, T - 1):
        b[i + 1] = omega + alpha * xi[i] * X[i - 1] / sigma**2 + beta * b[i]
        xi[i + 1] = (X[i + 1] - a - b[i + 1] * X[i])

    if visualization:
        plt.figure(tight_layout=True)
        plt.plot(b, linewidth=1, label='Filtered Data')
        plt.legend()
        plt.grid(True)
        plt.title(f'[omega, a , alpha, beta]: {res.x}')
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
                b, a, xi, res = estimation(
                    reduced_loglikelihood.loglikelihood, x, par[i], method='Nelder-Mead', targeting_estimation=args.targ_est, verbose=args.verbose, visualization=args.visualization)
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
        ax1.title.set_text('a')
        ax2 = plt.subplot(1, 5, 3)
        ax2.hist(omega_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax2.title.set_text('omega')
        ax3 = plt.subplot(1, 5, 4)
        ax3.hist(alpha_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax3.title.set_text('alpha')
        ax4 = plt.subplot(1, 5, 5)
        ax4.hist(beta_sgm, bins=40, color='blue',
                 edgecolor='darkblue', alpha=0.6)
        ax4.title.set_text('beta')
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
        X = np.load('/run/media/danielemdn/0A9A624E5CE1F5FA/saved_data/GAS100/dis_res100.npy')
        days = X.shape[0]
        n_stocks = X.shape[1]
        time_list = []
        day = np.random.randint(days)
        stock = np.random.randint(n_stocks)
        print(day, stock)
        x = X[day, stock, :]
        if args.targ_est:
            init_params = np.random.uniform(0, 0.5, size=3)
            b, a, xi, res = estimation(loglikelihood.loglikelihood_after_targ, x, init_params, targeting_estimation=args.targ_est, visualization=True)
        else:
            init_params = np.random.uniform(0, 0.5, size=4)
            b, a, xi, res = estimation(loglikelihood.loglikelihood, x, init_params, targeting_estimation=args.targ_est,verbose=args.verbose, visualization=True)
            print(ML_errors(res.x, x, b))
        if (math.isnan(b[-1])) or (b[-1] < 0):
            print(b[-1])
