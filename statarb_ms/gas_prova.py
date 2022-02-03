from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from tqdm import tqdm
import time


def gas_poisson_generate(w, A, B, size, dynamics):
    y = np.zeros(shape=size)
    f = np.zeros(shape=size)
    ###############################################
    # Initial conditions for the recursion:
    ###############################################
    y[0] = np.random.poisson(f[0])
    ###############################################
    # GAS mechanism:
    ###############################################
    for t in range(size - 1):
        if dynamics == 'gas':
            f[t+1] = w + A * (y[t]-np.exp(f[t]))*(np.exp(f[t])) + B * f[t]
        if dynamics == 'sin':
            f[t + 1] = 0.5 * np.sin(np.pi * t / 200)
        if dynamics == 'step':
            if t < 300: f[t + 1] = 0.1
            if t >= 300: f[t + 1] = 1
            if t >= 600: f[t + 1] = 0.1
        if dynamics == 'exp':
            f[t + 1] = np.exp(-(t + 1)/100)
        y[t + 1] = np.random.poisson(np.exp(f[t + 1]))

    return y, f

def gas_poisson(q, y):
    T = y.shape[0]
    A, B, w = q[0], q[1], q[2]
    f = np.zeros_like(y)
    for t in range(T - 1):
        f[t + 1] = w + A * (y[t] - np.exp(f[t])) * (np.exp(f[t])) + B * f[t]

    sum = 0
    for t in range(T):
        sum = sum + poisson.logpmf(y[t], mu=np.exp(f[t]))

    return - sum / y.shape[0]


if __name__ == '__main__':
    plt.style.use('seaborn')
    np.random.seed(666)

    A = 0.081
    B = -0.395
    w = 0.183
    estimates = np.zeros(shape=3)
    n = 1000
    # Generate Data
    y, f = gas_poisson_generate(w, A, B, size=1000, dynamics='exp')
    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(12, 5))
    axs[0].plot(y[:n], linewidth=1, label='Real')
    axs[0].set_ylabel('X')
    axs[1].plot(f[:n], 'k', linewidth=1, label='Real')
    axs[1].set_ylabel('f')

    # Fit a GAS model on the Data
    init_params = np.random.uniform(0, 0.1, size=3)
    res = minimize(gas_poisson, init_params, y, method='BFGS')
    std_err = np.sqrt([res.hess_inv[i, i] * (1 / y.shape[0])
                      for i in range(init_params.shape[0])])
    par_est = res.x
    par_est_up = res.x + std_err
    par_est_down = res.x - std_err

    print('True values: ', [A, B, w])
    print('Estimated values: ', par_est)
    print('Standard errors: ', std_err)

    # Filtered Data
    f_est = np.zeros_like(y)
    f_est_up = np.zeros_like(y)
    f_est_down = np.zeros_like(y)

    for t in range(f_est.shape[0] - 1):
        f_est[t + 1] = par_est[2] + par_est[0] * \
            (y[t] - np.exp(f_est[t])) * \
            (np.exp(f_est[t])) + par_est[1] * f_est[t]
        f_est_up[t + 1] = par_est_up[2] + par_est_up[0] * \
            (y[t] - np.exp(f_est_up[t])) * \
            (np.exp(f_est_up[t])) + par_est_up[1] * f_est_up[t]
        f_est_down[t + 1] = par_est_down[2] + par_est_down[0] * (y[t] - np.exp(
            f_est_down[t])) * (np.exp(f_est_down[t])) + par_est_down[1] * f_est_down[t]
    axs[1].plot(f_est[:n], 'crimson', linewidth=1, label='Filtered')
    axs[1].fill_between(list(range(n)), f_est_down[:n],
                        f_est_up[:n], color='crimson', alpha=0.4)
    axs[1].legend()
    plt.show()
