import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from makedir import go_up
import argparse
import logging
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as mpatches
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

def plot_returns(name1, name2):
    ret = np.load(go_up(1) + f'/saved_data/{name1}.npy')
    spy_ret = np.load(go_up(1) + f'/saved_data/{name2}.npy')
    percs = np.load(go_up(1) + '/saved_data/perc_positions.npy')
    plt.figure()
    plt.plot(ret, 'k', linewidth=1, label='Strategy PnL', alpha=0.8)
    plt.plot(spy_ret, 'crimson', linewidth=1, alpha=0.7, label='Buy and hold PnL')
    plt.xticks(x_label_position, x_label_day, fontsize=13,  rotation=60)
    plt.grid(True)
    plt.legend(fontsize=13)
    plt.show()

    plt.rcParams.update(IPython_default)
    plt.figure()
    plt.fill_between(range(percs.shape[0]), 0, percs[:,2], color='gray', label='Closed')
    plt.fill_between(range(percs.shape[0]), percs[:,2], percs[:,2] + percs[:,1], color='crimson', label='Short')
    plt.fill_between(range(percs.shape[0]), percs[:,2] + percs[:,1], percs[:,2] + percs[:,1] + percs[:,0], color='green', label='Long')
    plt.xticks(x_label_position, x_label_day, fontsize=13,  rotation=60)
    plt.yticks(np.arange(0,1.2,0.2), ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=13)
    plt.legend(fontsize=13)
    plt.show()

def plot_bvalues(name):
    b_values = np.load(go_up(1) + f'/saved_data/{name}.npy')
    plt.figure()
    # 9 is the index for APPLE
    b, = plt.plot(b_values[:, 9, 0], 'k')
    plt.plot(b_values[:, 9, 1], 'crimson', linewidth=0.2)
    plt.plot(b_values[:, 9, 2], 'crimson', linewidth=0.2)
    plt.fill_between(range(b_values.shape[0]), b_values[:, 9, 1], b_values[:, 9, 2], color='crimson', alpha=0.2)
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    red_patch = mpatches.Patch(color='red')
    plt.legend(handles=[b, red_patch], labels=['Estimated b values', '95% confidence interval'], fontsize=13)
    plt.grid(True)
    plt.show()

def rsquared_statistics(name):
    R_squared = np.load(go_up(1) + f'/saved_data/{name}.npy')
    ax = sns.heatmap(R_squared.T, cmap='YlOrBr', cbar_kws={'label': 'Coefficient of determination'})
    plt.xticks(x_label_position, x_label_day, fontsize=13, rotation=60)
    plt.yticks(y_label_position, y_label_day, fontsize=13)
    plt.show()

    plt.figure()
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

    plt.figure()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-p", "--plots", action='store_true',
                        help=("Plot returns"))
    parser.add_argument('-m', '--merge', action='store_true',
                        help='Merge files outputed by scoring.py')
    parser.add_argument('-o', '--onefile', action='store_true',
                            help='Merge just one file. To be used after merge parser set to true')
    parser.add_argument('-b', '--bvalues', action='store_true',
                        help='After args.plot=True, choose to plot b_values')
    parser.add_argument('-r', '--ret', action='store_true',
                        help='After args.plot=True, choose to plot returns')
    parser.add_argument('-rsq', '--rsquared', action='store_true',
                        help='After args.plot=True, choose to plot heatmap and histogram for the coefficient of determination for AR(1) model')
    parser.add_argument('-nt', '--OU_res_normtest', action='store_true',
                        help="Plot heatmap and histogram for the p-values from a Pearson and D'Agostino normality test on AR(1) residuals")
    parser.add_argument('-lb', '--ljungbox_test', action='store_true',
                        help='Plot heatmap and histogram for the p-values from Ljung-Box test on on AR(1) residuals ')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    IPython_default = plt.rcParams.copy()
    plt.style.use('seaborn')
    trading_days = np.array(pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date)
    tickers = pd.read_csv(go_up(1) + '/saved_data/ReturnsData.csv').columns.to_list()

    if args.plots:
        if args.ret:
            name1 = input('Name of the returns file: ')
            name2 = input('Name of the SPY returns file: ')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(252, len(trading_days), x_quantity)
            x_label_day = [trading_days[i] for i in x_label_position]
            plot_returns(name1, name2)
        if args.bvalues:
            name = input('Name of the b values file: ')
            x_quantity = int(input('Step of the labels on x-axis: '))
            x_label_position = np.arange(252, len(trading_days), x_quantity)
            x_label_day = [trading_days[i] for i in x_label_position]
            plot_bvalues(name)
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


    if args.merge:
        num_processes = int(input('Enter the number of processes: '))
        # logging.warning('Assuming number of processes used is 5...')
        num_pid = int(input('First pid number: '))
        file_merge([num_pid+i for i in list(range(num_processes))], one_file=args.onefile)
