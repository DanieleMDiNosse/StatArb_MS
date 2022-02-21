import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from makedir import go_up
from data import get_returns
import argparse
import logging
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew
import matplotlib.patches as mpatches
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import os
import time
from regression_parameters import auto_regression

def plot_pnl(pnl, pnl_gas, pnl_gas1, spy_pnl, percs):
    plt.figure(figsize=(12,8), tight_layout=True)
    plt.plot(pnl, 'k', linewidth=1, label='Strategy PnL', alpha=0.8)
    plt.plot(pnl_gas, 'green', linewidth=1, label='GAS Strategy PnL (60 days)', alpha=0.8)
    plt.plot(pnl_gas1, 'purple', linewidth=1, label='GAS Strategy PnL (120 days)', alpha=0.8)
    plt.plot(spy_pnl, 'crimson', linewidth=1, alpha=0.7, label='Buy and Hold PnL')
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    plt.grid(True)
    plt.legend(fontsize=11, loc='lower right')
    plt.show()

    plt.rcParams.update(IPython_default)
    plt.figure(figsize=(10,8))
    plt.fill_between(range(percs.shape[0]), 0, percs[:,2], color='gray', label='Closed')
    plt.fill_between(range(percs.shape[0]), percs[:,2], percs[:,2] + percs[:,1], color='crimson', label='Short')
    plt.fill_between(range(percs.shape[0]), percs[:,2] + percs[:,1], percs[:,2] + percs[:,1] + percs[:,0], color='green', label='Long')
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    plt.yticks(np.arange(0,1.2,0.2), ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=11)
    plt.legend(fontsize=11, loc='lower right')
    plt.show()

def plot_returns(ret, ret_gas, spy_ret):
    ret_today, ret_gas_today, spy_ret_today = ret[1:], ret_gas[1:], spy_ret[1:]
    ret_yesterday, ret_gas_yesterday, spy_ret_yesterday = ret[:-1], ret_gas[:-1], spy_ret[:-1]
    ret, ret_gas, spy_ret = (ret_today - ret_yesterday)/ret_yesterday, (ret_gas_today - ret_gas_yesterday)/ret_gas_yesterday, (spy_ret_today - spy_ret_yesterday)/spy_ret_yesterday
    fig = plt.figure(figsize=(12,8))
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

    fig = plt.figure(figsize=(10,10))
    ax3 = plt.subplot(313)
    ax3.hist(spy_ret, bins=100, color='crimson', alpha=0.8)
    ax3.set_title('Buy and Hold Returns')
    ax3.text(0.038, 700, 'Mean: %.3f' %spy_ret.mean(), fontsize=11)
    ax3.text(0.038, 620, 'Std: %.3f' %spy_ret.std(), fontsize=11)
    ax3.text(0.038, 540, 'Skew: %.3f' %skew(spy_ret), fontsize=11)
    ax2 = plt.subplot(312, sharex=ax3)
    ax2.hist(ret_gas, color='green', bins=100, alpha=0.8)
    ax2.set_title('GAS Strategy Returns')
    plt.tick_params(labelbottom=False)
    ax2.text(0.038, 400, 'Mean: %.3f' %ret_gas.mean(), fontsize=11)
    ax2.text(0.038, 350, 'Std: %.3f' %ret_gas.std(), fontsize=11)
    ax2.text(0.038, 300, 'Skew: %.3f' %skew(ret_gas), fontsize=11)
    ax1 = plt.subplot(311, sharex=ax3)
    ax1.hist(ret, color='k', bins=100, alpha=0.8)
    ax1.set_title('Strategy Returns')
    plt.tick_params(labelbottom=False)
    ax1.text(0.038, 400, 'Mean: %.3f' %ret.mean(), fontsize=11)
    ax1.text(0.038, 350, 'Std: %.3f' %ret.std(), fontsize=11)
    ax1.text(0.038, 300, 'Skew: %.3f' %skew(ret), fontsize=11)
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
            par, pred, resid, conf_int = auto_regression(X[t:t+est_window])
            b_list.append(par[1])
            pred_list.append(pred)
            resid_list.append(resid)
            plus_list.append(conf_int[1][1])
            minus_list.append(conf_int[1][0])
        plt.figure(figsize=(12,8), tight_layout=True)
        plt.plot(b_list, 'k', linewidth=1)
        plt.fill_between(list(range(len(b_list))), plus_list, minus_list, color='crimson', alpha=0.2)

    b_values = np.load(go_up(1) + f'/saved_data/{name}.npy')
    plt.figure(figsize=(12,8))
    # 9 is the index for APPLE
    b, = plt.plot(b_values[:, 9, 0], 'k', linewidth=1)
    plt.fill_between(range(b_values.shape[0]), b_values[:, 9, 1], b_values[:, 9, 2], color='crimson', alpha=0.2)
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    red_patch = mpatches.Patch(color='red')
    plt.legend(handles=[b, red_patch], labels=['Estimated b values', '95% confidence interval'], fontsize=13)
    plt.show()

def rsquared_statistics(name):
    R_squared = np.load(go_up(1) + f'/saved_data/{name}.npy')
    ax = sns.heatmap(R_squared.T, cmap='YlOrBr', cbar_kws={'label': 'Coefficient of determination'})
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    plt.yticks(y_label_position, y_label_day, fontsize=13)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.hist(R_squared.flatten(), bins=20, color='k')
    plt.xlabel('Coefficient of determination')
    plt.yscale('log')
    plt.grid()
    plt.show()

def binary(x, threshold=0.05):
    if x > threshold:
        x = 1
    else:
        x = 0
    return x

def normtest_discreteOU(name):
    bin_vec = np.vectorize(binary)
    normtest = bin_vec(np.load(go_up(1) + f'/saved_data/{name}.npy'))
    ax = sns.heatmap(normtest.T, cmap=['darkred', 'darkorange'])
    colorbar = ax.collections[0].colorbar
    ones = normtest.flatten().sum()/normtest.flatten().shape[0]
    zeros = 1 - ones
    colorbar.set_ticks(np.array([0,zeros,ones]))
    colorbar.set_ticklabels(['Rejected','Accepted'])
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    plt.yticks(y_label_position, y_label_day, fontsize=13)
    plt.show()

    plt.figure(figsize=(12,8))
    plt.hist(normtest.flatten(), bins=20, color='k')
    plt.grid(True)
    plt.show()

def ljung_box_test(name, order):

    if order == 1:
        dis_res = np.load(go_up(1) + f'/saved_data/{name}.npy')
    if order == 2:
        dis_res = (np.load(go_up(1) + f'/saved_data/{name}.npy'))**2

    lb_test = np.zeros(shape=(dis_res.shape[0], dis_res.shape[1]))
    bin_vec = np.vectorize(binary)
    for day in tqdm(range(dis_res.shape[0])):
        for stock in range(dis_res.shape[1]):
            lb_test[day, stock] = acorr_ljungbox(dis_res[day,stock,:], lags=2)[1][1] # p-values for lag=2

    lb_test = bin_vec(lb_test)
    ones = lb_test.flatten().sum()/lb_test.flatten().shape[0]
    zeros = 1 - ones
    ax = sns.heatmap(lb_test.T, cmap=['darkred', 'darkorange'])
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    plt.yticks(y_label_position, y_label_day, fontsize=13)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0,zeros,ones]))
    colorbar.set_ticklabels(['Rejected','Accepted'])
    plt.title(f'Accepted: {ones}, Rejected: {zeros}')
    plt.show()

def stationarity_check(name_residuals_file):
    '''Augumented Dickey Fuller test in order to check the null hypothesis that there is a unit root in the time series. This function, so,test
    the stationarity of the residuals of the regression of stocks on the risk factors. The function plot a heatmap illustrating the fraction of times that the null hypothesis is accepted (p-value > 0.05)
    or rejected (p-values < 0.05).'''

    residuals = np.load(go_up(1) + f'/saved_data/{name_residuals_file}.npy')
    days = residuals.shape[0]
    n_stocks = residuals.shape[1]
    p_values = np.zeros(shape=(days, n_stocks))
    bin_vec = np.vectorize(binary)
    for day in tqdm(range(1)):
        for stock in range(1):
            res = adfuller(residuals[day, stock, :])
            p_values[day, stock] = res[1]
    p_values = bin_vec(p_values)
    ones = p_values.flatten().sum()/p_values.flatten().shape[0]
    zeros = 1 - ones
    ax = sns.heatmap(p_values.T, cmap=['darkred', 'darkorange'])
    plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=12)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0,zeros,ones]))
    colorbar.set_ticklabels(['Rejected','Accepted'])
    plt.title(f'Accepted: {ones}, Rejected: {zeros}')
    plt.show()

def file_merge(pidnums, file_list):

    for file in file_list:
        try:
            df_score = [pd.read_csv(go_up(1) + f'/saved_data/{file}_{i}.csv') for i in pidnums]
            name = input('Name for the Score csv file: ')
            pd.concat(df_score, ignore_index=True).to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)
        except:
            splitted_files = [np.load(go_up(1) + f'/saved_data/{file}_{i}.npy') for i in pidnums]
            name = input(f'Name for the {file} file: ')
            np.save(go_up(1) + f'/saved_data/{name}', np.vstack(splitted_files))

def remove_file(pidnums, file_list):
    for file in file_list:
        try:
            [os.remove(go_up(1) + f'/saved_data/{file}_{i}.csv') for i in pidnums]
        except:
            [os.remove(go_up(1) + f'/saved_data/{file}_{i}.npy') for i in pidnums]

def sharpe_ratio(pnl, benchmark_pnl):
    pnl_ret = get_returns(pd.DataFrame(pnl), export_returns_csv=False)
    benchmark_pnl_ret = get_returns(pd.DataFrame(benchmark_pnl), export_returns_csv=False)
    diff = pnl_ret - benchmark_pnl_ret
    sharpe_ratio = diff.mean()/diff.std() * 100
    return sharpe_ratio


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
    parser.add_argument('-rsq', '--rsquared', action='store_true',
                        help='After args.plot=True, choose to plot heatmap and histogram for the coefficient of determination for AR(1) model.')
    parser.add_argument('-nt', '--OU_res_normtest', action='store_true',
                        help="Plot heatmap and histogram for the p-values from a Pearson and D'Agostino normality test on AR(1) residuals.")
    parser.add_argument('-lb', '--ljungbox_test', action='store_true',
                        help='Plot heatmap for the p-values from Ljung-Box test on on AR(1) residuals.')
    parser.add_argument('-s', '--stationarity', action='store_true',
                        help='Plot heatmap for p-values from Augumented Dickey Fuller test on residuals. ')

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
    trading_days = np.array(pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date)
    tickers = pd.read_csv(go_up(1) + '/saved_data/ReturnsData.csv').columns.to_list()

    if args.plots:
        if args.pnl:
            name1 = input('Name of the standard pnl file: ')
            name2 = input('Name of gas pnl file: ')
            name4 = input('Name of gas pnl file (1): ')
            name3 = input('Name of the SPY pnl file: ')
            pnl = np.load(go_up(1) + f'/saved_data/{name1}.npy')
            pnl_gas = np.load(go_up(1) + f'/saved_data/{name2}.npy')
            pnl_gas1 = np.load(go_up(1) + f'/saved_data/{name4}.npy')
            spy_pnl = np.load(go_up(1) + f'/saved_data/{name3}.npy')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(0, len(trading_days) - 252, x_quantity)
            x_label_day = [trading_days[252 + i] for i in x_label_position]
            name5 = input('Name of the positions percentage file: ')
            percs = np.load(go_up(1) + f'/saved_data/{name5}.npy')
            plot_pnl(pnl, pnl_gas, pnl_gas1, spy_pnl, percs)
        if args.ret:
            plot_returns(pnl, pnl_gas, spy_pnl)
            name1 = input('Name of the standard pnl file: ')
            name2 = input('Name of gas pnl file: ')
            name3 = input('Name of the SPY pnl file: ')
            pnl = np.load(go_up(1) + f'/saved_data/{name1}.npy')
            pnl_gas = np.load(go_up(1) + f'/saved_data/{name2}.npy')
            spy_pnl = np.load(go_up(1) + f'/saved_data/{name3}.npy')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(0, len(trading_days) - 252, x_quantity)
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
            normtest_discreteOU(name)
    if args.OU_res_normtest:
        name = input('Name of the p-values file: ')
        x_quantity = int(input('Step of the labels on x-axis: '))
        y_quantity = int(input('Step of the labels on y-axis: '))
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        normtest_discreteOU(name)

    if args.ljungbox_test:
        name = input('Name of the residuals file: ')
        order = int(input('Simple autocorrelation (1) or autocorellation of the squares (2)?:'))
        x_quantity = int(input('Step of the labels on x-axis: '))
        y_quantity = int(input('Step of the labels on y-axis: '))
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        ljung_box_test(name, order)

    if args.stationarity:
        name = input('Name of the residuals file: ')
        x_quantity = int(input('Step of the labels on x-axis: '))
        y_quantity = int(input('Step of the labels on y-axis: '))
        x_label_position = np.arange(252, len(trading_days), x_quantity)
        x_label_day = [trading_days[i] for i in x_label_position]
        y_label_position = np.arange(0, len(tickers), y_quantity)
        y_label_day = [tickers[i] for i in y_label_position]
        stationarity_check(name)


    if args.merge:
        num_processes = int(input('Enter the number of processes: '))
        # logging.warning('Assuming number of processes used is 5...')
        num_pid = int(input('First pid number: '))
        file_merge([num_pid+i for i in list(range(num_processes))], one_file=args.onefile)
