# import numpy as np
import argparse
import logging
import math
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import telegram_send
from data import price_data
from factors import money_on_stock, pca, risk_factors
from gas import estimation
from loglikelihood import (loglikelihood, loglikelihood_after_targ,
                           targeting_loglikelihood)
from makedir import go_up
from post_processing import LM_test_statistic, file_merge, remove_file
from regression_parameters import auto_regression, regression
from scipy.stats import normaltest
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def generate_data(df_returns, n_factor=15, lookback_for_factors=252, lookback_for_residual=60, export=True):
    '''This function uses an amount of days equal to lookback_for_factors to evaluate the PCA components and an amount equal to lookback_for_residual
    to evaluate the parameters of the Ornstein-Uhlenbeck process for the residuals. The output is composed by the dataframe of s_score for each stock
    and the beta factors.

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns for each stock.
    n_factor : int
        Number of principal components to keep in the PCA algorithm. The default is 15.
    lookback_for_factors : int
        Number of days used for the PCA. The default is 252 (one year of trade).
    lookback_for_residual : inf
        Number of days used for the estimation of the residual process (OU). The default is 60.
    export : bool
        Choose whether or not to export to csv the s_score dataframe and to .npy the vector of betas.

    Returns
    -------
    X : numpy ndarray
        Discrete version of the Ornstein Uhlenbeck process. This array will be used to estimate s-scores.
    beta_tensor : numpy ndarray
        Weights of each of the pca components for each day and each stock. Dimensions are (trading days x n_stocks x n_factors).
    Q : numpy ndarray
        Money to invest on each factors for each days. Dimensions are dim (trading days x n_factors x n_components x n_stocks)
    '''

    sp500cost = pd.read_pickle(
        go_up(1) + '/saved_data/SP500histcost_matched.pkl')
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    # I create some empy file named as the process PID, in order to use them to merge all the files (via file_merge function) that the different processes will create.
    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days), desc=f'{os.getpid()}'):
        # A time window of 252 days (one trading year) is shifted by 1 day at every iteration. Every time a PCA is
        # done over such period and the risk factors are evaluated, together with the residual process.
        # [0,252[, [1,253[, [2,254[ etc ->

        # I consider the 252 days slice of both the costituents and returns dataframe
        oneyear = sp500cost.iloc[i:lookback_for_factors + i]
        period = df_returns[i:lookback_for_factors + i]

        # Then I check for all the different lists of tickers in the slice in the costituents dataframe in the slice.
        # I add all the tickers present in the aforementioned lists in alloneyearticks list.
        # Since there will be a lot of repetitions, I consider only the unique tickers.
        oneyearticks = []
        alloneyearticks = np.unique(oneyear['Tickers'])
        for l_ticks in alloneyearticks:
            for tick in l_ticks:
                oneyearticks.append(tick)
        oneyearticks = list(set(oneyearticks))

        # Now that I have collected all the tickers that appear in the slice of 252 days, I have to select the columns in the slice
        # of the returns dataframe whose tick is present in oneyearticks.
        # Then I drop both the missing values and the all-zeros columns.
        period = period[period.columns.intersection(oneyearticks)]
        period = period.dropna(axis=1)
        period = period.loc[:, (period != 0).any(axis=0)]

        # PCA on such period
        eigenvalues, eigenvectors = pca(period, n_components=n_factor)

        # # (shape Q -> trading_days x n_factors x n_stocks)
        # For each day out the 252 ones, I evaluate the amount that has to be invested in each company into each factors.
        # In order to do so, I firstly need to know the indexes of the companies with respect to df_returns.columns, as I need to
        # assign the right positions to the Qij, since it is created as a unique numpy array for the whole time horizon.
        idxs = [df_returns.columns.get_loc(
            period.columns[i]) for i in range(period.shape[1])]
        # money on stock shape -> n_factors, n_stocks
        Q[i][:, idxs] = money_on_stock(period, eigenvectors)

        # Risk factors returns.
        factors = risk_factors(period, Q[i][:, idxs], eigenvectors)

        # Finally the residual process can be estimated.
        for stock in period.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            beta0, betas, conf_inter, residuals, pred, _ = regression(
                factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])  # regression(exog, endog) ovvero regression(X,Y)

            # Tested multiple time if beta_tensor and Q are well matched. They are.
            beta_tensor[i, stock_idx, :] = betas
            res[i, stock_idx, :] = residuals

            X = np.cumsum(residuals)
            dis_res[i, stock_idx, :] = X

    if export:
        np.save(
            f'/mnt/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
        np.save(f'/mnt/saved_data/Q_{os.getpid()}', Q)
        np.save(f'/mnt/saved_data/dis_res_{os.getpid()}', dis_res)
        np.save(f'/mnt/saved_data/res_{os.getpid()}', res)


def only_scoring(dis_res, df_returns, method, lookback_for_factors, lookback_for_residual, targeting_estimation=False):

    if targeting_estimation:
        num_par = 3
    else:
        num_par = 4

    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    score = np.zeros(shape=(trading_days, n_stocks))
    estimates = np.zeros(shape=(trading_days, n_stocks, num_par))

    # I initialize some counters.
    c, cc, ccc = 0, 0, 0

    # I create some empty file named as the process PID, in order to use them to merge all the files (via file_merge function) that the different processes will create.
    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days), desc=f'{os.getpid()}'):
        for stock in df_returns.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            X = dis_res[i, stock_idx, :]

            # Avoid estimation on zero residuals
            if X.any(axis=0) == 0:
                cc += 1
                continue

            if method == 'constant_speed':
                X = np.append(X, X[-1])
                parameters, discrete_pred, xi, discrete_conf_int = auto_regression(
                    X)
                a, b = parameters[0], parameters[1]

            if method == 'gas_modelization':
                if targeting_estimation:
                    b, a, xi, est, std = estimation(
                        X, targeting_estimation=targeting_estimation)
                else:
                    b, a, xi, est = estimation(X)
                estimates[i, stock_idx, :] = est.x
                b = b[-1]

                if b < 0:
                    X = np.append(X, X[-1])
                    parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(
                        X)
                    a, b = parameters[0], parameters[1]
                    c += 1

            # if b == 0.0:
            #     cc += 1
            #     continue
            # else:
            k = -np.log(b) * lookback_for_factors
            if k < lookback_for_factors / (0.5 * lookback_for_residual):
                score[i, stock_idx] = 0
                ccc += 1
            else:
                m = a / (1 - b)
                sgm_eq = np.std(xi) * np.sqrt(1 / (1 - b**2))
                score[i, stock_idx] = -m / sgm_eq

    telegram_send.send(messages=[
                       f'Number of negative b values for process {os.getpid()}: {c}', f'Number of stock with speed of mean reversion refused for process {os.getpid()}: {ccc}', f'Number of zero b for process {os.getpid()}: {cc} out of {n_stocks * trading_days}'])

    df_score = pd.DataFrame(score, columns=df_returns.columns)

    if method == 'gas_modelization':
        df_score.to_pickle(f'/mnt/saved_data/df_score_gas_{os.getpid()}.pkl')
        np.save(f'/mnt/saved_data/estimates_{os.getpid()}', estimates)

    if method == 'constant_speed':
        df_score.to_pickle(f'/mnt/saved_data/df_score_{os.getpid()}.pkl')


def SPY_beta(df_returns, spy, lookback_for_factors=252, lookback_for_residual=60, export=True):

    df_returns = np.array(df_returns)
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    vec_beta_spy = np.zeros(shape=(trading_days, n_stocks))
    for i in tqdm(range(trading_days), desc='SPY'):
        period = df_returns[i:lookback_for_factors + i]
        spy_period = spy[i:lookback_for_factors + i]

        for stock in range(df_returns.shape[1]):
            beta0, betas, conf_intervals, residuals, predictions, rsquared = regression(
                spy_period[-lookback_for_residual:], period[-lookback_for_residual:, stock])
            vec_beta_spy[i, stock] = betas[0]
    if export:
        name = input('Name of the file that will be saved: ')
        np.save(go_up(1) + f'/saved_data/{name}', vec_beta_spy)

    return vec_beta_spy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-s", "--scoring", action='store_true',
                        help='If passed, compute only the s-scores, provided residuals are already been computed.')
    parser.add_argument("-g", "--gas", action='store_true',
                        help=("Use gas estimation for the mean reverting speed. The default is False."))
    parser.add_argument("-sp", "--spy", action='store_true',
                        help=("Scoring for SPY."))
    parser.add_argument('-r', '--range', action='store_true',
                        help='Select a specific time range between 1995-01-03 to 2020-12-31')
    parser.add_argument("-t", "--targ_est", action='store_true')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    start = time.time()

    df_returns = pd.read_pickle("/mnt/saved_data/ReturnsDataHugeCleaned.pkl")[:4030]
    # df_returns = pd.read_pickle("/mnt/saved_data/ReturnsDataHugeCleaned.pkl")[:1643]

    if args.spy:
        spy = pd.read_csv(go_up(1) + "/saved_data/spy.csv")
        SPY_beta(df_returns, spy)

    df = [df_returns[:750], df_returns[498:1249], df_returns[997:1748], df_returns[1496:2247],
          df_returns[1995:2746], df_returns[2494:3245], df_returns[2993:3744], df_returns[3492:]]

    # df = [df_returns[:425], df_returns[173:599], df_returns[347:773], df_returns[521:947],
    #       df_returns[695:1121], df_returns[869:1295], df_returns[1043:1469], df_returns[1217:]]

    if args.gas == True:
        method = 'gas_modelization'
    else:
        method = 'constant_speed'

    if args.range:
        start = str(input('Start date (YYY-MM-DD): '))
        end = str(input('End date (YYY-MM-DD): '))
        date = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv')
        start = date.index[date.Date == start].tolist()[0]
        end = date.index[date.Date == end].tolist()[0]
        print(start, end)
        df_returns = pd.read_csv(go_up(1) +
                                 "/saved_data/ReturnsData.csv")[start: end + 1]
        generate_data(df_returns, method)
        end = time.time()

    else:
        telegram_send.send(
            messages=[f'========== {time.asctime()} =========='])
        np.random.seed()

        if args.scoring:
            logging.info('Computing the s-scores...')
            time.sleep(0.3)
            name = input('Name of the dis_res file: ')
            dis_res = np.load(f"/mnt/saved_data/{name}.npy")

            dr = [dis_res[:750, :, :], dis_res[498:1249, :, :], dis_res[997:1748, :, :], dis_res[1496:2247, :, :],
                  dis_res[1995:2746, :, :], dis_res[2494:3245, :, :], dis_res[2993:3744, :, :], dis_res[3492:, :, :]]

            # dr = [dis_res[:425, :, :], dis_res[173:599, :, :], dis_res[347:773, :, :], dis_res[521:947, :, :],
            #       dis_res[695:1121, :, :], dis_res[869:1295, :, :], dis_res[1043:1469, :, :], dis_res[1217:, :, :]]
            time.sleep(1)
            processes = [mp.Process(target=only_scoring, args=(
                i, j, method, 252, 80)) for i, j in zip(dr, df)]

        else:
            logging.info('Estimating residual process...')
            time.sleep(0.3)
            processes = [mp.Process(target=generate_data, args=(
                i,15, 252, 80, True)) for i in df]

        os.system('rm tmp/*')
        for p in processes:
            p.start()
            time.sleep(0.5)
        for p in processes:
            p.join()
        end = time.time()

        pidnums = [int(x) for x in os.listdir('tmp')]
        pidnums.sort()
        if args.gas:
            if args.scoring:
                file_list = ['df_score']
            else:
                file_list = ['res', 'dis_res', 'Q', 'beta_tensor']
        else:
            if args.scoring:
                file_list = ['df_score']
            else:
                file_list = ['res', 'dis_res', 'Q', 'beta_tensor']
        logging.info('Merging files, then remove splitted ones...')
        file_merge(pidnums, file_list)
        remove_file(pidnums, file_list)
        os.system('rm tmp/*')
        telegram_send.send(messages=[f'==========================='])

    time_elapsed = (end - start)
    logging.info('Time required for generate s-scores: %.2f seconds' %
                 time_elapsed)
