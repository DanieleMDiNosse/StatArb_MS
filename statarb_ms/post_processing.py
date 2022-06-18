import argparse
import logging
import os
import random
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from data import get_returns
from regression_parameters import auto_regression
from scipy.stats import chi2, kstest, normaltest, shapiro, skew, ttest_1samp
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def rejected_stocks():
    subset = ['ORLY', 'JCI', 'WHR', 'NUE',
              'LHX', 'ABMD', 'BDX', 'D', 'SWN', 'WMB']
    score70gas_logistic = pd.read_pickle(
        '/mnt/saved_data/scores/ScoreDataGas70_volint_logistic_reduced.pkl')
    score70gas_logistic = score70gas_logistic.fillna(0)
    score70gas = pd.read_pickle(
        '/mnt/saved_data/scores/ScoreDataGas70_volint.pkl')[subset]

    # score80 = pd.read_pickle('/mnt/saved_data/scores/ScoreDataGas70_volint_logistic_reduced.pkl')[subset]
    # score90 = pd.read_pickle('/mnt/saved_data/scores/ScoreDataGas70_volint_logistic_reduced.pkl')[subset]

    score70 = pd.read_pickle(
        '/mnt/saved_data/scores/ScoreData70_volint.pkl')[subset]
    # scoregas80 = pd.read_pickle('/mnt/saved_data/scores/ScoreDataGas80_volint.pkl')[subset]
    # scoregas90 = pd.read_pickle('/mnt/saved_data/scores/ScoreDataGas90_volint.pkl')[subset]

    score_dataframe_list = [score70, score70gas, score70gas_logistic]

    counts = np.zeros(shape=len(score_dataframe_list))
    for df, i in zip(score_dataframe_list, range(len(score_dataframe_list))):
        count = 0
        for col in df.columns:
            try:
                count += df[col].value_counts()[0]
            except:
                pass
        counts[i] = count / (df.shape[0] * df.shape[1]) * 100

    plt.figure(figsize=(12, 8), tight_layout=True)
    x = np.arange(0, len(score_dataframe_list))
    labels = [r'$\tilde{T}=70$', r'$\tilde{T}=70$', r'$\tilde{T}=70$']
    colors = ['crimson', 'crimson', 'crimson', 'blue', 'blue', 'blue']
    colors = ['crimson', 'blue', 'purple']
    edgecolors = ['darkred', 'darkblue', 'darkmagenta']
    plt.bar(x[0], counts[0], color=colors[0],
            edgecolor=edgecolors[0], alpha=0.6)
    plt.bar(x[1], counts[1], color=colors[1],
            edgecolor=edgecolors[1], alpha=0.6)
    plt.bar(x[2], counts[2], color=colors[2],
            edgecolor=edgecolors[2], alpha=0.6)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.grid(True)
    plt.ylabel(r'% of refused $\kappa$')
    plt.legend(['Constant b', r'Time-varying $b_t$ (Identity link function)',
               r'Time-varying $b_t$ (Logistic link function)'])
    plt.show()


def plot_returns(ret, ret_gas, spy_ret):
    ret_today, ret_gas_today, spy_ret_today = ret[1:], ret_gas[1:], spy_ret[1:]
    ret_yesterday, ret_gas_yesterday, spy_ret_yesterday = ret[:-
                                                              1], ret_gas[:-1], spy_ret[:-1]
    ret, ret_gas, spy_ret = (ret_today - ret_yesterday) / ret_yesterday, (ret_gas_today -
                                                                          ret_gas_yesterday) / ret_gas_yesterday, (spy_ret_today - spy_ret_yesterday) / spy_ret_yesterday
    fig = plt.figure(figsize=(12, 8))
    ax3 = plt.subplot(313)
    ax3.plot(spy_ret, 'crimson', linewidth=0.7, alpha=0.8)
    ax3.set_title('Buy and Hold Returns')
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    ax2 = plt.subplot(312, sharex=ax3)
    ax2.plot(ret_gas, 'green', linewidth=0.7, alpha=0.8)
    ax2.set_title('GAS Strategy Returns')
    plt.tick_params(labelbottom=False)
    ax1 = plt.subplot(311, sharex=ax3)
    ax1.plot(ret, 'k', linewidth=0.7, alpha=0.8)
    ax1.set_title('Strategy Returns')
    plt.tick_params(labelbottom=False)

    fig = plt.figure(figsize=(10, 10))
    ax3 = plt.subplot(313)
    ax3.hist(spy_ret, bins=100, color='crimson', alpha=0.8)
    ax3.set_title('Buy and Hold Returns')
    ax3.text(0.038, 700, 'Mean: %.3f' % spy_ret.mean(), fontsize=11)
    ax3.text(0.038, 620, 'Std: %.3f' % spy_ret.std(), fontsize=11)
    ax3.text(0.038, 540, 'Skew: %.3f' % skew(spy_ret), fontsize=11)
    ax2 = plt.subplot(312, sharex=ax3)
    ax2.hist(ret_gas, color='green', bins=100, alpha=0.8)
    ax2.set_title('GAS Strategy Returns')
    plt.tick_params(labelbottom=False)
    ax2.text(0.038, 400, 'Mean: %.3f' % ret_gas.mean(), fontsize=11)
    ax2.text(0.038, 350, 'Std: %.3f' % ret_gas.std(), fontsize=11)
    ax2.text(0.038, 300, 'Skew: %.3f' % skew(ret_gas), fontsize=11)
    ax1 = plt.subplot(311, sharex=ax3)
    ax1.hist(ret, color='k', bins=100, alpha=0.8)
    ax1.set_title('Strategy Returns')
    plt.tick_params(labelbottom=False)
    ax1.text(0.038, 400, 'Mean: %.3f' % ret.mean(), fontsize=11)
    ax1.text(0.038, 350, 'Std: %.3f' % ret.std(), fontsize=11)
    ax1.text(0.038, 300, 'Skew: %.3f' % skew(ret), fontsize=11)
    plt.show()


def plot_bvalues(name, synthetic_check=False):
    if synthetic_check:
        X = np.zeros(shape=5000)
        b_list, pred_list, resid_list, plus_list, minus_list = [], [], [], [], []
        a, b = 0.1, 0.5
        for t in range(0, X.shape[0] - 1):
            X[t + 1] = a + b * X[t] + np.random.normal()
        est_window = int(input('Length of the estimation window: '))
        for t in tqdm(range(0, X.shape[0] - est_window), desc='MLE on synthetic AR(1) data'):
            par, pred, resid, conf_int = auto_regression(X[t:t + est_window])
            b_list.append(par[1])
            pred_list.append(pred)
            resid_list.append(resid)
            plus_list.append(conf_int[1][1])
            minus_list.append(conf_int[1][0])
        plt.figure(figsize=(12, 8), tight_layout=True)
        plt.plot(b_list, 'k', linewidth=1)
        plt.fill_between(list(range(len(b_list))), plus_list,
                         minus_list, color='crimson', alpha=0.2)

    b_values = np.load(go_up(1) + f'/saved_data/{name}.npy')
    plt.figure(figsize=(12, 8), tight_layout=True)
    # 9 is the index for APPLE
    b, = plt.plot(b_values[:, 9, 0], 'k', linewidth=1)
    plt.fill_between(range(
        b_values.shape[0]), b_values[:, 9, 1], b_values[:, 9, 2], color='crimson', alpha=0.2)
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    red_patch = mpatches.Patch(color='red')
    plt.legend(handles=[b, red_patch], labels=[
               'Estimated b values', '95% confidence interval'], fontsize=13)
    plt.show()


def plot_alphas(name1, name2):
    alpha_values = np.load(go_up(1) + f'/saved_data/{name1}.npy')
    sgm_eq = np.load(go_up(1) + f'/saved_data/{name2}.npy')
    stock = np.random.randint(0, 364)
    plt.figure(figsize=(12, 8), tight_layout=True)
    plt.plot(alpha_values[:, stock], 'k', linewidth=1, label='alphas')
    plt.plot(sgm_eq[:, stock], 'b', linewidth=1, label='std_eq')
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.show()


def rsquared_statistics(name):
    R_squared = np.load(f'/mnt/saved_data/{name}.npy')
    ax = sns.heatmap(R_squared.T, cbar_kws={
                     'label': 'Coefficient of determination'})
    plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=12)
    plt.show()

    plt.figure(figsize=(12, 8), tight_layout=True)
    plt.hist(R_squared.flatten(), bins=20, color='blue', alpha=0.7)
    plt.xlabel('Coefficient of determination')
    plt.yscale('log')
    # plt.grid()
    plt.show()


def binary(x, threshold=0.05):
    if x > threshold:
        x = 1
    else:
        x = 0
    return x


def normtest_discreteOU():
    '''Perform D'Agostino and Pearson normality test'''

    subset = ['ORLY', 'JCI', 'WHR', 'NUE',
              'LHX', 'ABMD', 'BDX', 'D', 'SWN', 'WMB']
    ret = pd.read_pickle('/mnt/saved_data/returns/ReturnsVolData.pkl')
    stock_position = [np.where(ret.columns == col)[0][0] for col in subset]
    name_list = ['AR_res_gas70_volint_logistic_reduced']
    # name_list = ['AR_res_gas70_volint']
    # path = str(input('Path of residuals of AR(1) process: '))
    for name in name_list:
        logging.info(f'{name}')
        # [:, stock_position, :]
        xi = np.load(f'/mnt/saved_data/AR_res/{name}.npy')
        pvals = np.zeros(shape=(xi.shape[0], xi.shape[1]))

        for stock in tqdm(range(xi.shape[1])):
            for day in range(xi.shape[0]):
                statistic, pvalue = normaltest(xi[day, stock, :])
                pvals[day, stock] = pvalue

        bin_vec = np.vectorize(binary)
        normtest = bin_vec(pvals)
        ax = sns.heatmap(normtest.T,  cmap=['azure', 'black'])
        colorbar = ax.collections[0].colorbar
        ones = normtest.flatten().sum() / normtest.flatten().shape[0]
        zeros = 1 - ones
        colorbar.set_ticks(np.array([0, ones, zeros]))
        colorbar.set_ticklabels(['Not rejected', 'Rejected'])

        trading_days = np.array(pd.read_pickle(
            '/mnt/saved_data/PriceData.pkl').Date)[:4030 + 126]
        tickers = ret.columns[stock_position]
        x_quantity = 252
        y_quantity = 1
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
        plt.yticks(y_label_position, y_label_day, fontsize=12)
        plt.title(fr'$\alpha$ < 0.05: {zeros}, $\alpha$ > 0.05: {ones}')
        plt.tight_layout()
        # plt.savefig(f'../../NORMTEST_{name}.png')
        plt.show()


def ljung_box_test():

    subset = ['ORLY', 'JCI', 'WHR', 'NUE',
              'LHX', 'ABMD', 'BDX', 'D', 'SWN', 'WMB']
    ret = pd.read_pickle('/mnt/saved_data/returns/ReturnsVolData.pkl')
    stock_position = [np.where(ret.columns == col)[0][0] for col in subset]

    # name_list = ['AR_res_gas70_volint_logistic_reduced']
    name_list = ['AR_res_gas70_volint']
    order_list = [1, 2]
    for name in name_list:
        for order in order_list:
            logging.info(f'name: {name}, order: {order}')
            if order == 1:
                xi = np.load(
                    f'/mnt/saved_data/AR_res/{name}.npy')[:, stock_position, :]
            if order == 2:
                xi = (
                    np.load(f'/mnt/saved_data/AR_res/{name}.npy'))[:, stock_position, :]**2

            lb_test = np.zeros(shape=(xi.shape[0], xi.shape[1]))
            bin_vec = np.vectorize(binary)

            for day in tqdm(range(xi.shape[0])):
                for stock in range(xi.shape[1]):
                    lb_test[day, stock] = acorr_ljungbox(
                        xi[day, stock, :], lags=2, model_df=1, return_df=True)['lb_pvalue'][2]

            lb_test = bin_vec(lb_test)
            ones = lb_test.flatten().sum() / lb_test.flatten().shape[0]
            zeros = 1 - ones
            print(fr'{name} $\alpha$ < 0.05: {zeros}, $\alpha$ > 0.05: {ones}')
            # plt.figure(figsize=(10,8), tight_layout=True)
            # ax = sns.heatmap(lb_test.T,  cmap=['azure', 'black'])
            # plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            # plt.yticks(y_label_position, y_label_day, fontsize=12)
            # colorbar = ax.collections[0].colorbar
            # colorbar.set_ticks(np.array([0,zeros,ones]))
            # colorbar.set_ticklabels(['Rejected','Not rejected'])
            # plt.title(fr'{name} $\alpha$ < 0.05: {zeros}, $\alpha$ > 0.05: {ones}')
            # plt.savefig(f'../../{name}_{order}.png')
    plt.show()


def crosscorr(x, y, max_lag, bootstrap_test=False):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cross_corr = []
    for d in range(max_lag):
        cc = 0
        for i in range(len(x) - d):
            cc += (x[i] - x_mean) * (y[i + d] - y_mean)
        cc = cc / np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        cross_corr.append(cc)
    plt.plot(cross_corr, 'k')
    plt.title('Cross-correlation function')
    plt.xlabel('Lags')

    if bootstrap_test:
        cross_corr_s = []
        for i in range(100):
            xs, ys = x, y
            np.random.shuffle(xs)
            np.random.shuffle(ys)
            xs_mean = np.mean(xs)
            ys_mean = np.mean(ys)
            cross_corr_s_i = []
            for d in range(max_lag):
                cc = 0
                for i in range(len(x) - d):
                    cc += (xs[i] - xs_mean) * (ys[i + d] - ys_mean)
                cc = cc / np.sqrt(np.sum((xs - xs_mean)**2)
                                  * np.sum((ys - ys_mean)**2))
                cross_corr_s_i.append(cc)
            cross_corr_s.append(cross_corr_s_i)
        meancc = np.mean(np.array(cross_corr_s), axis=0)
        stdcc = np.std(np.array(cross_corr_s), axis=0)
        plt.plot(meancc - 3 * stdcc, 'crimson', lw=0.5)
        plt.plot(meancc, 'crimson', lw=0.5)
        plt.plot(meancc + 3 * stdcc, 'crimson', lw=0.5)
        plt.fill_between(np.arange(0, max_lag), meancc + 3 *
                         stdcc, meancc - 3 * stdcc, color='crimson', alpha=0.6)
    plt.grid()
    return cross_corr


def onesample_ttest(name):
    dis_res = np.load(go_up(1) + f'/saved_data/{name}.npy')
    t_test = np.zeros(shape=(dis_res.shape[0], dis_res.shape[1]))
    bin_vec = np.vectorize(binary)
    for day in tqdm(range(dis_res.shape[0])):
        for stock in range(dis_res.shape[1]):
            t_test[day, stock] = ttest_1samp(dis_res[day, stock], popmean=0)[
                1]  # p-values for lag=2

    t_test = bin_vec(t_test)
    ones = t_test.flatten().sum() / t_test.flatten().shape[0]
    zeros = 1 - ones
    ax = sns.heatmap(t_test.T,  cmap=['azure', 'black'])
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=11)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0, zeros, ones]))
    colorbar.set_ticklabels(['Rejected', 'Not rejected'])
    plt.title(f'Not rejected: {ones}, Rejected: {zeros}')
    plt.show()


def stationarity_check(name_residuals_file):
    '''Augumented Dickey Fuller test in order to check the null hypothesis that there is a unit root in the time series. This function, so, test
    the stationarity of the residuals of the regression of stocks on the risk factors. The function plot a heatmap illustrating the fraction of
    times that the null hypothesis is Not rejected (p-value > 0.05) or rejected (p-values < 0.05).'''

    residuals = np.load(f'/mnt/saved_data/{name_residuals_file}.npy')

    days = residuals.shape[0]
    n_stocks = residuals.shape[1]
    p_values = np.zeros(shape=(days, n_stocks))
    bin_vec = np.vectorize(binary)

    for day in tqdm(range(days)):
        for stock in range(n_stocks):
            res = adfuller(residuals[day, stock, :])
            p_values[day, stock] = res[1]

    p_values = bin_vec(p_values)
    ones = p_values.flatten().sum() / p_values.flatten().shape[0]
    zeros = 1 - ones
    ax = sns.heatmap(p_values.T, cmap=['azure', 'black'])

    plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=12)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0, zeros, ones]))
    colorbar.set_ticklabels(['Rejected', 'Not rejected'])
    plt.title(fr'{name} $\alpha$ < 0.05: {zeros}, $\alpha$ > 0.05: {ones}')
    plt.show()


def LM_test_statistic(X, params):
    '''Lagrange Multiplier test. It tests the presence of instantaneous autocorellation
    in the score under the null hypothesis of constant parameter.
    See "Iliyan Georgiev, David I. Harvey, Stephen J. Leybourne, A.M. Robert Taylor, Testing for parameter
    instability in predictive regression models, Journal of Econometrics, Volume 204, Issue 1, 2018."

    Parameters
    ----------
    X : numpy ndarray
        Time series over which the estimation of the GAS model has been done.
    params: list or numpy ndarray
        Maximum Likelihood estimates.

    Returns
    -------
    pvalue: float
        One side p-value.
    '''
    omega, a, sigma = params[0], params[1], params[-1]

    delta_w = X[1:-1] / sigma**2 * (X[2:] - a - omega * X[1:-1])
    scaled_score = delta_w * sigma / X[1:-1]
    delta_a = (X[2:] - a - X[1:-1] * omega) / sigma**2
    delta_al = delta_w / sigma**2 * \
        (X[1:-1]**2 * X[:-2] - a * X[:-2] * X[1:-1] - omega * X[:-2]**2 * X[1:-1])
    delta_b = delta_w * omega * X[1:-1]

    if sigma == 1:
        # regressors = np.array([delta_a[1:], delta_al[1:], delta_b[1:], delta_w[1:], scaled_score[:-1] * delta_w[1:]]).T
        regressors = np.array(
            [delta_a[1:], delta_w[1:], scaled_score[:-1] * delta_w[1:]]).T
    else:
        delta_sgm = - 1 / sigma + delta_w**2 * sigma
        regressors = np.array([delta_a[1:], delta_al[1:], delta_b[1:],
                              delta_sgm[1:], delta_w[1:], scaled_score[:-1] * delta_w[1:]]).T
        # regressors = np.array([delta_a[1:], delta_sgm[1:], delta_w[1:], scaled_score[:-1] * delta_w[1:]]).T

    response = np.ones_like(X[:-3]).T
    ones_est = np.zeros_like(response)[1:]
    model = sm.OLS(response[1:], regressors[1:])  # endog, exog
    res = model.fit()

    if sigma == 1:
        # c_a, c_al, c_b, c_w, c_A = res.params
        c_a, c_w, c_A = res.params
        ones_est = c_a * regressors[:, 0] + c_w * \
            regressors[:, 1] + c_A * regressors[:, 2]
    else:
        c_a, c_al, c_b, c_sgm, c_w, c_A = res.params
        # c_a, c_w, c_sgm, c_A = res.params
        ones_est = c_a * regressors[:, 0] + c_al * regressors[:, 1] + c_b * regressors[:,
                                                                                       2] + c_sgm * regressors[:, 3] + c_w * regressors[:, 4] + c_A * regressors[:, 5]
        # ones_est = c_a * regressors[:, 0] + c_sgm * regressors[:, 1] + c_w * regressors[:, 2] + c_A * regressors[:, 3]

    ESS = ((ones_est - response.mean())**2).sum()
    pvalue = 1 - chi2.cdf(ESS, df=1)

    return pvalue


def file_merge(pidnums, file_list):

    for file in file_list:
        try:
            df_score = [pd.read_pickle(
                f'/mnt/saved_data/{file}_{i}.pkl') for i in pidnums]
            name = input(f'Name for the {file} pkl file: ')
            pd.concat(df_score, ignore_index=True).to_pickle(
                f'/mnt/saved_data/{name}.pkl')
        except:
            splitted_files = [
                np.load(f'/mnt/saved_data/{file}_{i}.npy') for i in pidnums]
            name = input(f'Name for the {file} npy file: ')
            np.save(f'/mnt/saved_data/{name}', np.vstack(splitted_files))

    for file in file_list:
        try:
            [os.remove(f'/mnt/saved_data/{file}_{i}.pkl') for i in pidnums]
        except:
            [os.remove(f'/mnt/saved_data/{file}_{i}.npy') for i in pidnums]


def sharpe_ratio(pnl, benchmark_pnl, period):
    sharpe_ratio = []
    pnl_ret = get_returns(pd.DataFrame(pnl), export=False, m=1)
    benchmark_pnl_ret = get_returns(
        pd.DataFrame(benchmark_pnl), export=False, m=1)
    # max() is used to let the function handle a period of length equal to the maximum available, pnl.shape[0]

    for i in range(max(1, pnl.shape[0] - period)):
        pnl_ret_period = pnl_ret[i:i + period]
        benchmark_pnl_ret_period = benchmark_pnl_ret[i:i + period]
        diff = pnl_ret_period - benchmark_pnl_ret_period
        sharpe_ratio.append(diff.mean() / diff.std())

    return np.array(np.sqrt(252) * np.array(sharpe_ratio))


def yearly_returns():
    plt.style.use('seaborn')
    pnls80 = np.array(
        [np.load(f'../saved_data/PnL/pnl_LBFGSB(80days({i})).npy') for i in range(15)])
    plt.figure(figsize=(8, 6), tight_layout=True)
    yearly_rets = np.zeros(shape=(pnls80.shape[0], 15))

    for k in range(pnls80.shape[0]):
        pnl = pnls80[k]
        for i, j in zip(range(0, 3528, 252), range(15)):
            yearly_rets[k, j] = (pnl[i + 252] - pnl[i]) / pnl[i] * 100
        # last year is not totally complete
        yearly_rets[k, -1] = (pnl[-1] - pnl[-250]) / pnl[-250] * 100

    yearly_rets_mean = yearly_rets.mean(axis=0)
    yearly_rets_std = yearly_rets.std(axis=0)
    plt.bar(np.arange(15), yearly_rets_mean, color='black', alpha=0.8)
    plt.bar(np.arange(15), 2 * yearly_rets_std,
            bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    plt.bar(np.arange(15), - 2 * yearly_rets_std,
            bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (80 days)')

    pnls60 = np.array(
        [np.load(f'../saved_data/PnL/pnl_LBFGSB(60days({i})).npy') for i in range(15)])
    plt.figure(figsize=(8, 6), tight_layout=True)
    yearly_rets = np.zeros(shape=(pnls60.shape[0], 15))

    for k in range(pnls60.shape[0]):
        pnl = pnls60[k]
        for i, j in zip(range(0, 3528, 252), range(15)):
            yearly_rets[k, j] = (pnl[i + 252] - pnl[i]) / pnl[i] * 100
        # last year is not totally complete
        yearly_rets[k, -1] = (pnl[-1] - pnl[-250]) / pnl[-250] * 100

    yearly_rets_mean = yearly_rets.mean(axis=0)
    yearly_rets_std = yearly_rets.std(axis=0)
    plt.bar(np.arange(15), yearly_rets_mean, color='black', alpha=0.8)
    plt.bar(np.arange(15), 2 * yearly_rets_std,
            bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    plt.bar(np.arange(15), - 2 * yearly_rets_std,
            bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (60 days)')

    pnlAL60 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(60days).npy')
    plt.figure(figsize=(8, 6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)

    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL60[i + 252] - pnlAL60[i]) / pnlAL60[i] * 100

    yearly_rets[-1] = (pnlAL60[-1] - pnlAL60[-250]) / \
        pnlAL60[-250] * 100  # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (60 days)')

    pnlAL80 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(80days).npy')
    plt.figure(figsize=(8, 6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)
    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL80[i + 252] - pnlAL80[i]) / pnlAL80[i] * 100
    yearly_rets[-1] = (pnlAL80[-1] - pnlAL80[-250]) / \
        pnlAL80[-250] * 100  # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (80 days)')

    pnlAL100 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(100days).npy')
    plt.figure(figsize=(8, 6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)
    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL100[i + 252] - pnlAL100[i]) / pnlAL100[i] * 100
    yearly_rets[-1] = (pnlAL100[-1] - pnlAL100[-250]) / \
        pnlAL100[-250] * 100  # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (100 days)')
    plt.show()


def drift_vs_sgm():

    alpha_name = str(input('Name of the alphas file: '))
    sgm_eq_name = str(input('Name of the sgm_eq file: '))

    alphas = np.load(f'/mnt/saved_data/{alpha_name}.npy')
    sgm_eq = np.load(f'/mnt/saved_data/{sgm_eq_name}.npy')

    n_stocks = alphas.shape[1]

    tickers = pd.read_pickle(
        '/mnt/saved_data/ReturnsData.pkl').columns.to_list()
    stock_idx = random.randint(0, n_stocks)
    print(tickers[stock_idx])

    plt.style.use('seaborn')
    plt.figure(figsize=(10, 5), tight_layout=True)
    trading_days = np.array(pd.read_pickle(
        '/mnt/saved_data/PriceData.pkl').Date)[:4030 + 252]
    x_quantity = 126
    x_label_position = np.arange(252, len(trading_days), x_quantity)
    x_label_day = [trading_days[i] for i in x_label_position]
    plt.xticks(np.arange(0, len(trading_days) - x_quantity * 3,
               x_quantity), x_label_day[:-1], fontsize=12, rotation=90)
    plt.plot(alphas[:, stock_idx] / 252, 'red', linewidth=1.2,
             alpha=0.8, label=r'Drift $\alpha$')
    plt.plot(sgm_eq[:, stock_idx], 'green', linewidth=1.2,
             alpha=0.8, label=r'Std $\tilde{\sigma}^{eq}$')
    plt.legend()
    plt.show()


def beta():

    name = str(input('Name of strategy PnL: '))
    lookback = int(input('Lookback for beta computation: '))
    pnl_strat = np.load(f'/mnt/saved_data/PnL/{name}.npy')
    ret_strat = np.diff(pnl_strat)
    ret_first_comp = np.diff(np.load('/mnt/saved_data/PnL/pnl_firstcomp.npy'))

    betas = np.zeros(shape=pnl_strat.shape[0] - (lookback + 1))

    for i in range(betas.shape[0]):
        betas[i] = np.cov(ret_strat[i:i + lookback], ret_first_comp[i:i +
                          lookback])[0][1] / np.var(ret_first_comp[i:i + lookback])

    plt.style.use('seaborn')
    plt.figure(figsize=(12, 8), tight_layout=True)
    trading_days = np.array(pd.read_pickle(
        '/mnt/saved_data/PriceData.pkl').Date)[lookback:4030 + 126]
    tickers = pd.read_pickle(
        '/mnt/saved_data/returns/ReturnsData.pkl').columns.to_list()
    x_quantity = 126
    x_label_position = np.arange(252, len(trading_days), x_quantity)
    x_label_day = [trading_days[i] for i in x_label_position]
    plt.xticks(np.arange(0, len(trading_days) - x_quantity * 3,
               x_quantity), x_label_day[:-1], fontsize=12, rotation=90)
    plt.plot(betas, 'k', linewidth=1.2, alpha=0.8)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-p", "--plots", action='store_true',
                        help=("Plot returns."))
    parser.add_argument('-m', '--merge', action='store_true',
                        help='Merge files outputed by scoring.py.')
    parser.add_argument('-o', '--onefile', action='store_true',
                        help='Merge just one file. To be used after merge parser set to true.')
    parser.add_argument('-b', '--bvalues', action='store_true',
                        help='After args.plot=True, choose to plot b_values.')
    parser.add_argument('-pl', '--pnl', action='store_true',
                        help='After args.plot=True, choose to plot pnl')
    parser.add_argument('-r', '--ret', action='store_true',
                        help='After args.plot=True, choose to plot returns')
    parser.add_argument('-a', '--alphas', action='store_true',
                        help='After args.plot=True, choose to plot alphas vs std_eq')
    parser.add_argument('-rsq', '--rsquared', action='store_true',
                        help='After args.plot=True, choose to plot heatmap and histogram for the coefficient of determination for AR(1) model.')
    parser.add_argument('-nt', '--OU_res_normtest', action='store_true',
                        help="Plot heatmap and histogram for the p-values from a Pearson and D'Agostino normality test on AR(1) residuals.")
    parser.add_argument('-lb', '--ljungbox_test', action='store_true',
                        help='Plot heatmap for the p-values from Ljung-Box test on on AR(1) residuals.')
    parser.add_argument('-s', '--stationarity', action='store_true',
                        help='Plot heatmap for p-values from Augumented Dickey Fuller test on residuals. ')
    parser.add_argument('-t', '--ttest', action='store_true',
                        help='Plot heatmap for p-values from one sample t-test on the mean of the residuals. ')
    parser.add_argument('-rj', '--rej_stocks', action='store_true',
                        help='Bar plot for the percentage of stocks with mean reversion speed less than half period of estimation')
    parser.add_argument('-sr', '--sharpe_ratio', action='store_true',
                        help='Rolling sharpe ratio')
    parser.add_argument("-lm", "--lmtest", action='store_true', help='LM test')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    IPython_default = plt.rcParams.copy()
    plt.style.use('seaborn')
    np.random.seed(666)
    trading_days = np.array(pd.read_pickle(
        '/mnt/saved_data/PriceData.pkl').Date)[:4030 + 126]
    tickers = pd.read_pickle(
        '/mnt/saved_data/returns/ReturnsData.pkl').columns.to_list()

    if args.plots:
        if args.pnl:
            name1 = input(
                'Name of the standard pnl file (Avellaneda & Lee (60 days)): ')
            name11 = input(
                'Name of the standard pnl file (Avellaneda & Lee (100 days)): ')
            name2 = input(
                'Name of gas pnl file (GAS Strategy PnL (60 days run 1)): ')
            name4 = input(
                'Name of gas pnl file (GAS Strategy PnL (60 days run 2)): ')
            name6 = input(
                'Name of gas pnl file (GAS Strategy PnL (60 days run 3)): ')
            name8 = input(
                'Name of gas pnl file (GAS Strategy PnL (60 days w targ)): ')
            name7 = input(
                'Name of gas pnl file (GAS Strategy PnL (70 days)): ')
            name5 = input(
                'Name of gas pnl file (GAS Strategy PnL (100 days)): ')
            name9 = input(
                'Name of gas pnl file (GAS Strategy PnL (60 days bfgs)): ')
            name10 = input(
                'Name of gas pnl file (GAS Strategy PnL (80 days bfgs)): ')
            name12 = input(
                'Name of gas pnl file (GAS Strategy PnL (80 days bfgs(1))): ')
            name3 = input('Name of the SPY pnl file: ')
            pnl_gas1 = np.load(go_up(1) + f'/saved_data/PnL/{name5}.npy')
            pnl = np.load(
                go_up(1) + f'/saved_data/PnL/{name1}.npy')[:pnl_gas1.shape[0]]
            pnl_1 = np.load(go_up(1) + f'/saved_data/PnL/{name11}.npy')
            pnl_gas = np.load(
                go_up(1) + f'/saved_data/PnL/{name2}.npy')[:pnl_gas1.shape[0]]
            pnl_gas2 = np.load(go_up(1) + f'/saved_data/PnL/{name6}.npy')
            pnl_gas3 = np.load(go_up(1) + f'/saved_data/PnL/{name7}.npy')
            pnl_gas4 = np.load(go_up(1) + f'/saved_data/PnL/{name4}.npy')
            pnl_gas5 = np.load(go_up(1) + f'/saved_data/PnL/{name8}.npy')
            pnl_gas6 = np.load(go_up(1) + f'/saved_data/PnL/{name9}.npy')
            pnl_gas7 = np.load(go_up(1) + f'/saved_data/PnL/{name10}.npy')
            pnl_gas8 = np.load(go_up(1) + f'/saved_data/PnL/{name12}.npy')
            spy_pnl = np.load(
                go_up(1) + f'/saved_data/PnL/{name3}.npy')[:pnl_gas1.shape[0]]
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(
                0, len(trading_days) - 252, x_quantity)
            x_label_day = [trading_days[252 + i] for i in x_label_position]
            plot_pnl(pnl, pnl_1, pnl_gas, pnl_gas1, pnl_gas2, pnl_gas3,
                     pnl_gas4, pnl_gas5, pnl_gas6, pnl_gas7, pnl_gas8, spy_pnl)

        if args.ret:
            plot_returns(pnl, pnl_gas, spy_pnl)
            name1 = input('Name of the standard pnl file: ')
            name2 = input('Name of gas pnl file: ')
            name3 = input('Name of the SPY pnl file: ')
            pnl = np.load(go_up(1) + f'/saved_data/{name1}.npy')
            pnl_gas = np.load(go_up(1) + f'/saved_data/{name2}.npy')
            pnl = np.load(go_up(1) + f'/saved_data/{name1}.npy')
            spy_pnl = np.load(go_up(1) + f'/saved_data/{name3}.npy')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(
                0, len(trading_days) - 252, x_quantity)
            x_label_day = [trading_days[252 + i] for i in x_label_position]

        if args.bvalues:
            name = input('Name of the b values file: ')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(252, len(trading_days), x_quantity)
            x_label_day = [trading_days[i] for i in x_label_position]
            plot_bvalues(name, synthetic_check=True)

        if args.rsquared:
            name = input('Name of the R squared file: ')
            x_quantity = int(input('Step of the labels on x-axis: '))
            y_quantity = int(input('Step of the labels on y-axis: '))
            x_label_position = np.arange(252, len(trading_days), x_quantity)
            x_label_day = [trading_days[i] for i in x_label_position]
            y_label_position = np.arange(0, len(tickers), y_quantity)
            y_label_day = [tickers[i] for i in y_label_position]
            rsquared_statistics(name)

        if args.alphas:
            name1 = input('Name of the alpha values file: ')
            name2 = input('Name of the std_eq file: ')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(252, len(trading_days), x_quantity)
            x_label_day = [trading_days[i] for i in x_label_position]
            plot_alphas(name1, name2)

        if args.sharpe_ratio:
            logging.info('Rolling Sharpe Ratios')
            period = 252

            pnls80 = [
                np.load(f'../saved_data/PnL/pnl_LBFGSB(80days({i})).npy') for i in range(15)]
            pnls60 = [
                np.load(f'../saved_data/PnL/pnl_LBFGSB(60days({i})).npy') for i in range(15)]
            pnlAL60 = np.load(
                f'../saved_data/PnL/pnl_AvellanedaLee(60days).npy')[:pnls80[0].shape[0]]
            pnlAL80 = np.load(
                f'../saved_data/PnL/pnl_AvellanedaLee(80days).npy')
            pnlAL100 = np.load(
                f'../saved_data/PnL/pnl_AvellanedaLee(100days).npy')
            spy = np.load(
                '../saved_data/PnL/pnl_FirstPrincipalComp().npy')[:pnls80[0].shape[0]]

            SR80 = np.array([sharpe_ratio(pnl, spy, period) for pnl in pnls80])
            SR80_mean = SR80.mean(axis=0).flatten()
            SR80_std = SR80.std(axis=0).flatten()
            plt.figure(figsize=(12, 5), tight_layout=True)
            plt.plot(SR80_mean, 'k', linewidth=1)
            plt.fill_between(np.arange(
                SR80_mean.shape[0]), SR80_mean + 2 * SR80_std, SR80_mean - 2 * SR80_std, color='crimson', alpha=0.5)
            x_label_position = np.arange(252 * 2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (80 days)')

            SR60 = np.array([sharpe_ratio(pnl, spy, period) for pnl in pnls60])
            SR60_mean = SR60.mean(axis=0).flatten()
            SR60_std = SR60.std(axis=0).flatten()
            plt.figure(figsize=(12, 5), tight_layout=True)
            plt.plot(SR60_mean, 'k', linewidth=1)
            plt.fill_between(np.arange(
                SR60_mean.shape[0]), SR60_mean + 2 * SR60_std, SR60_mean - 2 * SR80_std, color='crimson', alpha=0.5)
            x_label_position = np.arange(252 * 2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (60 days)')

            plt.figure(figsize=(12, 5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL60, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252 * 2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (Avellaneda & Lee 60 days)')

            plt.figure(figsize=(12, 5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL80, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252 * 2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (Avellaneda & Lee 80 days)')

            plt.figure(figsize=(12, 5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL100, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252 * 2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (Avellaneda & Lee 100 days)')
            plt.show()

    if args.rej_stocks:
        name1 = input('Score dataframe (Avellaneda & Lee): ')
        name2 = input('Score dataframe (60 days): ')
        name3 = input('Score dataframe (60 days 1): ')
        name4 = input('Score dataframe (70 days): ')
        name5 = input('Score dataframe (100 days): ')
        name6 = input('Score dataframe (60 days w targ): ')
        names = [name1, name2, name3, name4, name5, name6]
        score_dataframe_list = [pd.read_csv(
            go_up(1) + f'/saved_data/{name}.csv') for name in names]
        rejected_stocks(score_dataframe_list)

    if args.OU_res_normtest:
        print('Normality Test')
        print("Sto dormendo finché non finisce l'amico mio...")
        time.sleep(1200)
        normtest_discreteOU()

    if args.ljungbox_test:
        # name = input('Name of the residuals file: ')
        # order = int(input('Simple autocorrelation (1) or autocorellation of the squares (2)?:'))
        # x_quantity = int(input('Step of the labels on x-axis: '))
        # y_quantity = int(input('Step of the labels on y-axis: '))
        x_quantity = 252
        y_quantity = 30
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        # ljung_box_test(name, order)
        ljung_box_test()

    if args.ttest:
        name = input('Name of the residuals file: ')
        x_quantity = int(input('Step of the labels on x-axis: '))
        y_quantity = int(input('Step of the labels on y-axis: '))
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        onesample_ttest(name)

    if args.stationarity:
        name = input('Name of the residuals file: ')
        x_quantity = int(input('Step of the labels on x-axis: '))
        y_quantity = int(input('Step of the labels on y-axis: '))
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        stationarity_check(name)

    if args.lmtest:
        # dis_res = input('Path of the residuals file: ')
        # estimates = input('Path of the estimates file: ')
        # X = np.load(dis_res)
        # params = np.load(estimates)
        estimates = np.load(
            '/mnt/saved_data/estimates_gas/estimates_gas90_volint.npy')
        X = np.load('/mnt/saved_data/AR_res/AR_res_gas90_volint.npy')
        p_values = np.zeros(shape=(X.shape[0], X.shape[1]))
        bin_vec = np.vectorize(binary)

        for day in tqdm(range(X.shape[0])):
            for stock in range(X.shape[1]):
                x = X[day, stock, :]
                params = estimates[day, stock, :]
                p_values[day, stock] = LM_test_statistic(x, params)

        p_values = bin_vec(p_values)
        ones = p_values.flatten().sum() / p_values.flatten().shape[0]
        zeros = 1 - ones

        x_quantity = 252
        y_quantity = 30
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]

        ax = sns.heatmap(p_values.T, cmap=['azure', 'black'])
        plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
        plt.yticks(y_label_position, y_label_day, fontsize=12)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(np.array([0, zeros, ones]))
        colorbar.set_ticklabels(['Rejected', 'Not rejected'])
        plt.title(fr'$\alpha$ < 0.05: {zeros}, $\alpha$ > 0.05: {ones}')
        plt.show()

    if args.merge:
        num_processes = int(input('Enter the number of processes: '))
        # logging.warning('Assuming number of processes used is 5...')
        num_pid = int(input('First pid number: '))
        file_merge([num_pid + i for i in list(range(num_processes))],
                   one_file=args.onefile)
