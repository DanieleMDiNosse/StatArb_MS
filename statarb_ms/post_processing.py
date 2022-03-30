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
from scipy.stats import ttest_1samp
import matplotlib.patches as mpatches
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import os
import time
from regression_parameters import auto_regression

def plot_pnl(pnl, pnl_1, pnl_gas, pnl_gas1, pnl_gas2, pnl_gas3, pnl_gas4, pnl_gas5, pnl_gas6, pnl_gas7, pnl_gas8, spy_pnl):

    plt.plot(pnl, 'k', linewidth=1, label='Avellaneda & Lee PnL (60 days)', alpha=0.8)
    plt.plot(pnl_gas, 'green', linewidth=1, label='GAS Strategy PnL (60 days run 1)', alpha=0.8)
    plt.plot(pnl_gas2, 'blue', linewidth=1, label='GAS Strategy PnL (60 days run 2)', alpha=0.8)
    plt.plot(pnl_gas4, 'gold', linewidth=1, label='GAS Strategy PnL (60 days run 3)', alpha=0.8)
    plt.plot(pnl_gas5, 'dodgerblue', linewidth=1, label='GAS Strategy PnL (60 days w targ)', alpha=0.8)
    plt.plot(pnl_gas6, linewidth=1, label='GAS Strategy PnL (60 days w bfgs)', alpha=0.8)
    plt.plot(pnl_gas3, 'orange', linewidth=1, label='GAS Strategy PnL (70 days)', alpha=0.8)
    plt.plot(pnl_gas7, linewidth=1, label='GAS Strategy PnL (80 days w bfgs)', alpha=0.8)
    plt.plot(pnl_gas8, linewidth=1, label='GAS Strategy PnL (80 days w bfgs(1))', alpha=0.8)
    plt.plot(pnl_1, linewidth=1, label='Avellaneda & Lee PnL (100 days)', alpha=0.8)
    plt.plot(pnl_gas1, 'purple', linewidth=1, label='GAS Strategy PnL (100 days)', alpha=0.8)
    plt.plot(spy_pnl, 'crimson', linewidth=1, alpha=0.7, label='Buy and Hold PnL')
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    plt.grid(True)
    plt.legend(fontsize=11, loc='upper left')
    plt.show()

def plot_raw_pnl():
    plt.style.use('seaborn')
    pnl_names = [str.split('.')[0] for str in os.listdir('../saved_data/PnL')]
    pnls = [np.load(go_up(1) + f'/saved_data/PnL/{name}.npy') for name in pnl_names]
    pnl_names = [str.split('_')[1] for str in pnl_names]
    plt.figure(figsize=(12,8), tight_layout=True)

    for pnl, pnl_name in zip(pnls, pnl_names):
        if (pnl_name.split('(')[1] == '60days') or (pnl_name.split('(')[1] == '60days)'):
            plt.plot(pnl[:4030-252], 'k', linewidth=1, label=f'{pnl_name}', alpha=0.4)
        if (pnl_name.split('(')[1] == '80days') or (pnl_name.split('(')[1] == '80days)'):
            plt.plot(pnl[:4030-252], 'b', linewidth=1, label=f'{pnl_name}', alpha=0.4)
        if (pnl_name.split('(')[1] == '100days') or (pnl_name.split('(')[1] == '100days)'):
            plt.plot(pnl[:4030-252], 'gray', linewidth=1, label=f'{pnl_name}', alpha=0.6)
        if pnl_name.split('(')[0] == 'AvellanedaLee':
            plt.plot(pnl[:4030-252], 'purple', linewidth=2.5, label=f'{pnl_name}', alpha=1)
        if pnl_name == 'FirstPrincipalComp()':
            plt.plot(pnl[:4030-252], 'crimson', linewidth=1, label=f'{pnl_name}', alpha=1)

    trading_days = np.array(pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date)[:4030+126]
    x_quantity = 126
    x_label_position = np.arange(0, len(trading_days) - 252, x_quantity)
    x_label_day = [trading_days[252 + i] for i in x_label_position]
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    plt.grid(True)
    # plt.legend(fontsize=11, loc='upper left')
    plt.show()

def rejected_stocks():
    score60 = [pd.read_csv(f'../saved_data/ScoreData_bfgs60_{i}.csv') for i in range(15)]
    score80 = [pd.read_csv(f'../saved_data/ScoreData_bfgs80_{i}.csv') for i in range(15)]
    scoreAL60 = pd.read_csv(f'../saved_data/ScoreData60.csv')
    scoreAL80 = pd.read_csv(f'../saved_data/ScoreData60.csv')
    scoreAL100 = pd.read_csv(f'../saved_data/ScoreData60.csv')
    score_dataframe_list = [scoreAL60, scoreAL80, scoreAL100]
    for i in score60:
        score_dataframe_list.append(i)
    for i in score80:
        score_dataframe_list.append(i)
    counts = np.zeros(shape=len(score_dataframe_list))
    for df, i in zip(score_dataframe_list, range(len(score_dataframe_list))):
        count = 0
        for col in df.columns:
            count += df[col].value_counts()[0]
        counts[i] = count/(df.shape[0]*df.shape[1]) * 100
    plt.figure(figsize=(12,8), tight_layout=True)
    x = np.arange(0, len(score_dataframe_list))
    label0 = [f'GAS(60days)_{i}' for i in range(len(score60))]
    label1 = [f'GAS(80days)_{i}' for i in range(len(score80))]
    label = ['A&L(60days)', 'A&L(80days)', 'A&L(100days)']
    for i in label0:
        label.append(i)
    for i in label1:
        label.append(i)
    print(label)
    colors = ['crimson','crimson','crimson','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray','gray', 'blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue']
    edgecolors = ['darkred','darkred','darkred','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black', 'darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue','darkblue',]
    plt.bar(x, counts, color=colors, edgecolor=edgecolors, alpha=0.6)
    plt.xticks(np.arange(len(label)), label, rotation=90)
    plt.grid(True)
    plt.ylabel('Percentage')
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
    plt.figure(figsize=(12,8), tight_layout=True)
    # 9 is the index for APPLE
    b, = plt.plot(b_values[:, 9, 0], 'k', linewidth=1)
    plt.fill_between(range(b_values.shape[0]), b_values[:, 9, 1], b_values[:, 9, 2], color='crimson', alpha=0.2)
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    red_patch = mpatches.Patch(color='red')
    plt.legend(handles=[b, red_patch], labels=['Estimated b values', '95% confidence interval'], fontsize=13)
    plt.show()

def plot_alphas(name1, name2):
    alpha_values = np.load(go_up(1) + f'/saved_data/{name1}.npy')
    sgm_eq = np.load(go_up(1) + f'/saved_data/{name2}.npy')
    stock = np.random.randint(0, 364)
    plt.figure(figsize=(12,8), tight_layout=True)
    plt.plot(alpha_values[:, stock], 'k', linewidth=1, label='alphas')
    plt.plot(sgm_eq[:, stock], 'b', linewidth=1, label='std_eq')
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    plt.legend()
    plt.grid(True)
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
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=13)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0,zeros,ones]))
    colorbar.set_ticklabels(['Rejected','Accepted'])
    plt.title(f'Accepted: {ones}, Rejected: {zeros}')
    plt.show()

def onesample_ttest(name):
    dis_res = np.load(go_up(1) + f'/saved_data/{name}.npy')
    t_test = np.zeros(shape=(dis_res.shape[0], dis_res.shape[1]))
    bin_vec = np.vectorize(binary)
    for day in tqdm(range(dis_res.shape[0])):
        for stock in range(dis_res.shape[1]):
            t_test[day, stock] = ttest_1samp(dis_res[day, stock], popmean=0)[1] # p-values for lag=2

    t_test = bin_vec(t_test)
    ones = t_test.flatten().sum()/t_test.flatten().shape[0]
    zeros = 1 - ones
    ax = sns.heatmap(t_test.T, cmap=['darkred', 'darkorange'])
    plt.xticks(x_label_position, x_label_day, fontsize=11, rotation=90)
    plt.yticks(y_label_position, y_label_day, fontsize=11)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.array([0,zeros,ones]))
    colorbar.set_ticklabels(['Rejected','Accepted'])
    plt.title(f'Accepted: {ones}, Rejected: {zeros}')
    plt.show()

def stationarity_check(name_residuals_file):
    '''Augumented Dickey Fuller test in order to check the null hypothesis that there is a unit root in the time series. This function, so, test
    the stationarity of the residuals of the regression of stocks on the risk factors. The function plot a heatmap illustrating the fraction of
    times that the null hypothesis is accepted (p-value > 0.05) or rejected (p-values < 0.05).'''

    residuals = np.load(go_up(1) + f'/saved_data/{name_residuals_file}.npy')
    days = residuals.shape[0]
    days = 500
    n_stocks = residuals.shape[1]
    p_values = np.zeros(shape=(days, n_stocks))
    bin_vec = np.vectorize(binary)
    for day in tqdm(range(days)):
        for stock in range(n_stocks):
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

def file_merge(pidnums, file_list, file_name):

    for file in file_list:
        try:
            df_score = [pd.read_csv(go_up(1) + f'/saved_data/{file}_{i}.csv') for i in pidnums]
            # name = input('Name for the Score csv file: ')
            pd.concat(df_score, ignore_index=True).to_csv(go_up(1) + f'/saved_data/{file_name}.csv', index=False)
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

def sharpe_ratio(pnl, benchmark_pnl, period):
    sharpe_ratio = np.zeros(shape=(pnl.shape[0] - period))
    pnl_ret = get_returns(pd.DataFrame(pnl), export_returns_csv=False, m=1)
    benchmark_pnl_ret = get_returns(pd.DataFrame(benchmark_pnl), export_returns_csv=False, m=1)
    for i in range(pnl.shape[0] - period):
        pnl_ret_period = pnl_ret[i:i+period]
        benchmark_pnl_ret_period = benchmark_pnl_ret[i:i+period]
        diff = pnl_ret_period - benchmark_pnl_ret_period
        sharpe_ratio[i] = diff.mean()/diff.std()
    return np.sqrt(period) * sharpe_ratio

def yearly_returns():
    plt.style.use('seaborn')
    pnls80 = np.array([np.load(f'../saved_data/PnL/pnl_LBFGSB(80days({i})).npy') for i in range(15)])
    plt.figure(figsize=(8,6), tight_layout=True)
    yearly_rets = np.zeros(shape=(pnls80.shape[0], 15))
    for k in range(pnls80.shape[0]):
        pnl = pnls80[k]
        for i, j in zip(range(0, 3528, 252), range(15)):
            yearly_rets[k, j] = (pnl[i + 252] - pnl[i]) / pnl[i] * 100
        yearly_rets[k, -1] = (pnl[-1] - pnl[-250])/pnl[-250] * 100 # last year is not totally complete
    yearly_rets_mean = yearly_rets.mean(axis=0)
    yearly_rets_std = yearly_rets.std(axis=0)
    plt.bar(np.arange(15), yearly_rets_mean, color='black', alpha=0.8)
    plt.bar(np.arange(15), 2 * yearly_rets_std, bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    plt.bar(np.arange(15), - 2 * yearly_rets_std, bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (80 days)')

    pnls60 = np.array([np.load(f'../saved_data/PnL/pnl_LBFGSB(60days({i})).npy') for i in range(15)])
    plt.figure(figsize=(8,6), tight_layout=True)
    yearly_rets = np.zeros(shape=(pnls60.shape[0], 15))
    for k in range(pnls60.shape[0]):
        pnl = pnls60[k]
        for i, j in zip(range(0, 3528, 252), range(15)):
            yearly_rets[k, j] = (pnl[i + 252] - pnl[i]) / pnl[i] * 100
        yearly_rets[k, -1] = (pnl[-1] - pnl[-250])/pnl[-250] * 100 # last year is not totally complete
    yearly_rets_mean = yearly_rets.mean(axis=0)
    yearly_rets_std = yearly_rets.std(axis=0)
    plt.bar(np.arange(15), yearly_rets_mean, color='black', alpha=0.8)
    plt.bar(np.arange(15), 2 * yearly_rets_std, bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    plt.bar(np.arange(15), - 2 * yearly_rets_std, bottom=yearly_rets_mean, color='crimson', alpha=0.5)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (60 days)')

    pnlAL60 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(60days).npy')
    plt.figure(figsize=(8,6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)
    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL60[i + 252] - pnlAL60[i]) / pnlAL60[i] * 100
    yearly_rets[-1] = (pnlAL60[-1] - pnlAL60[-250])/pnlAL60[-250] * 100 # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (60 days)')

    pnlAL80 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(80days).npy')
    plt.figure(figsize=(8,6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)
    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL80[i + 252] - pnlAL80[i]) / pnlAL80[i] * 100
    yearly_rets[-1] = (pnlAL80[-1] - pnlAL80[-250])/pnlAL80[-250] * 100 # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (80 days)')


    pnlAL100 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(100days).npy')
    plt.figure(figsize=(8,6), tight_layout=True)
    yearly_rets = np.zeros(shape=15)
    for i, j in zip(range(0, 3528, 252), range(15)):
        yearly_rets[j] = (pnlAL100[i + 252] - pnlAL100[i]) / pnlAL100[i] * 100
    yearly_rets[-1] = (pnlAL100[-1] - pnlAL100[-250])/pnlAL100[-250] * 100 # last year is not totally complete
    plt.bar(np.arange(15), yearly_rets, color='black', alpha=0.8)
    xlabel = [str(1996 + i) for i in range(15)]
    plt.xticks(np.arange(15), xlabel, rotation=90)
    plt.ylabel('%')
    plt.title('Yearly returns (100 days)')
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
    trading_days = np.array(pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date)[:4030 + 126]
    tickers = pd.read_csv(go_up(1) + '/saved_data/ReturnsData.csv').columns.to_list()

    if args.plots:
        if args.pnl:
            name1 = input('Name of the standard pnl file (Avellaneda & Lee (60 days)): ')
            name11 = input('Name of the standard pnl file (Avellaneda & Lee (100 days)): ')
            name2 = input('Name of gas pnl file (GAS Strategy PnL (60 days run 1)): ')
            name4 = input('Name of gas pnl file (GAS Strategy PnL (60 days run 2)): ')
            name6 = input('Name of gas pnl file (GAS Strategy PnL (60 days run 3)): ')
            name8 = input('Name of gas pnl file (GAS Strategy PnL (60 days w targ)): ')
            name7 = input('Name of gas pnl file (GAS Strategy PnL (70 days)): ')
            name5 = input('Name of gas pnl file (GAS Strategy PnL (100 days)): ')
            name9 = input('Name of gas pnl file (GAS Strategy PnL (60 days bfgs)): ')
            name10 = input('Name of gas pnl file (GAS Strategy PnL (80 days bfgs)): ')
            name12 = input('Name of gas pnl file (GAS Strategy PnL (80 days bfgs(1))): ')
            name3 = input('Name of the SPY pnl file: ')
            pnl_gas1 = np.load(go_up(1) + f'/saved_data/PnL/{name5}.npy')
            pnl = np.load(go_up(1) + f'/saved_data/PnL/{name1}.npy')[:pnl_gas1.shape[0]]
            pnl_1 = np.load(go_up(1) + f'/saved_data/PnL/{name11}.npy')
            pnl_gas = np.load(go_up(1) + f'/saved_data/PnL/{name2}.npy')[:pnl_gas1.shape[0]]
            pnl_gas2 = np.load(go_up(1) + f'/saved_data/PnL/{name6}.npy')
            pnl_gas3 = np.load(go_up(1) + f'/saved_data/PnL/{name7}.npy')
            pnl_gas4 = np.load(go_up(1) + f'/saved_data/PnL/{name4}.npy')
            pnl_gas5 = np.load(go_up(1) + f'/saved_data/PnL/{name8}.npy')
            pnl_gas6 = np.load(go_up(1) + f'/saved_data/PnL/{name9}.npy')
            pnl_gas7 = np.load(go_up(1) + f'/saved_data/PnL/{name10}.npy')
            pnl_gas8 = np.load(go_up(1) + f'/saved_data/PnL/{name12}.npy')
            spy_pnl = np.load(go_up(1) + f'/saved_data/PnL/{name3}.npy')[:pnl_gas1.shape[0]]
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(0, len(trading_days) - 252, x_quantity)
            x_label_day = [trading_days[252 + i] for i in x_label_position]
            # name5 = input('Name of the positions percentage file: ')
            # percs = np.load(go_up(1) + f'/saved_data/{name5}.npy')
            plot_pnl(pnl, pnl_1, pnl_gas, pnl_gas1, pnl_gas2, pnl_gas3, pnl_gas4, pnl_gas5, pnl_gas6, pnl_gas7, pnl_gas8, spy_pnl)
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

            pnls80 = [np.load(f'../saved_data/PnL/pnl_LBFGSB(80days({i})).npy') for i in range(15)]
            pnls60 = [np.load(f'../saved_data/PnL/pnl_LBFGSB(60days({i})).npy') for i in range(15)]
            pnlAL60 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(60days).npy')[:pnls80[0].shape[0]]
            pnlAL80 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(80days).npy')
            pnlAL100 = np.load(f'../saved_data/PnL/pnl_AvellanedaLee(100days).npy')
            spy = np.load('../saved_data/PnL/pnl_FirstPrincipalComp().npy')[:pnls80[0].shape[0]]

            SR80 = np.array([sharpe_ratio(pnl, spy, period) for pnl in pnls80])
            SR80_mean = SR80.mean(axis=0)
            SR80_std = SR80.std(axis=0)
            plt.figure(figsize=(12,5), tight_layout=True)
            plt.plot(SR80_mean, 'k', linewidth=1)
            plt.fill_between(np.arange(SR80_mean.shape[0]), SR80_mean + 2 * SR80_std, SR80_mean - 2 * SR80_std, color='crimson', alpha=0.5)
            x_label_position = np.arange(252*2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (80 days)')

            SR60 = np.array([sharpe_ratio(pnl, spy, period) for pnl in pnls60])
            SR60_mean = SR60.mean(axis=0)
            SR60_std = SR60.std(axis=0)
            plt.figure(figsize=(12,5), tight_layout=True)
            plt.plot(SR60_mean, 'k', linewidth=1)
            plt.fill_between(np.arange(SR60_mean.shape[0]), SR60_mean + 2 * SR60_std, SR60_mean - 2 * SR80_std, color='crimson', alpha=0.5)
            x_label_position = np.arange(252*2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (60 days)')

            plt.figure(figsize=(12,5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL60, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252*2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (Avellaneda & Lee 60 days)')

            plt.figure(figsize=(12,5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL80, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252*2, len(trading_days), 126)
            x_label_day = [trading_days[i] for i in x_label_position]
            plt.xticks(x_label_position, x_label_day, fontsize=12, rotation=90)
            plt.title('Rolling Sharpe Ratio (Avellaneda & Lee 80 days)')

            plt.figure(figsize=(12,5), tight_layout=True)
            plt.plot(sharpe_ratio(pnlAL100, spy, period), 'k', linewidth=1)
            x_label_position = np.arange(252*2, len(trading_days), 126)
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
        score_dataframe_list = [pd.read_csv(go_up(1) + f'/saved_data/{name}.csv') for name in names]
        rejected_stocks(score_dataframe_list)

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


    if args.merge:
        num_processes = int(input('Enter the number of processes: '))
        # logging.warning('Assuming number of processes used is 5...')
        num_pid = int(input('First pid number: '))
        file_merge([num_pid+i for i in list(range(num_processes))], one_file=args.onefile)
