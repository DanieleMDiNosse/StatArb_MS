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
from factors import money_on_stock, pca, risk_factors
from gas import estimation
from data import price_data
from loglikelihood import (loglikelihood, loglikelihood_after_targ,
                           targeting_loglikelihood)
from makedir import go_up
from post_processing import LM_test_statistic, file_merge, remove_file
from regression_parameters import auto_regression, regression
from scipy.stats import normaltest
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def generate_data(df_returns, n_factor, method, targeting_estimation, lookback_for_factors=252, lookback_for_residual=60, export=True):
    '''This function uses an amount of days equal to lookback_for_factors to evaluate the PCA components and an amount equal to lookback_for_residual
    to evaluate the parameters of the Ornstein-Uhlenbeck process for the residuals. The output is composed by the dataframe of s_score for each stock
    and the beta factors.

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns for each stock.
    n_factor : int
        Number of principal components to keep in the PCA algorithm.
    variable_n_factor : bool
        to be set to True if the number of principal components is chosen through a fixed amount of explained variance, False if it is set by n_factor.
        If it is True, n_factor can be any value: the PCA will not use it.
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
    constituents = pd.read_pickle(go_up(1) + '/saved_data/SP500histcost.pkl')

    trading_days = df_returns.shape[0] - lookback_for_factors  # 6294
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    beta_tensor = [[range(trading_days)], [], []]
    alpha_values = np.zeros(shape=(trading_days, n_stocks))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    sgm_eq = np.zeros(shape=(trading_days, n_stocks))
    if targeting_estimation:
        num_par = 3
    else:
        num_par = 4
    estimates = np.zeros(shape=(trading_days, n_stocks, num_par))

    # Il primo score corrisponderà al 252° giorno (indice 251)
    score = np.zeros(shape=(trading_days, n_stocks))
    b_values = np.zeros(shape=(trading_days, n_stocks, 3))
    b_values_gas = np.zeros(
        shape=(trading_days, n_stocks, lookback_for_residual))
    a_values_gas = np.zeros(shape=(trading_days, n_stocks, 1))
    R_squared = np.zeros(shape=(trading_days, n_stocks))
    dis_res_reg = np.zeros(
        shape=(trading_days, n_stocks, lookback_for_residual))
    c = 0  # counter to track the number of negative bs comes from GAS model
    # counter to track the number of k < lookback_for_factors / (0.5 * lookback_for_residual):
    cc = 0

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days), desc=f'{os.getpid()}'):
        s = time.time()
        start = constituents['Date'][i]
        end = constituents['Date'][lookback_for_factors + i]
        cost = constituents['Tickers'][i: lookback_for_factors + i]
        ticks = []
        for l in cost:
            for ticker in l:
                ticks.append(ticker)
        ticks = np.unique(np.array(ticks))
        prices = price_data(ticks, start, end, export_csv=False)
        df_returns = get_returns(prices, export_csv=False)
        f = time.time()
        print(f-s)
        time.sleep(2)

        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        # [0,252[, [1,253[ ecc -> ogni period comprende un anno di trading (252 giorni)
        period = df_returns[i:lookback_for_factors + i]
        eigenvalues, eigenvectors = pca(period, n_components=n_factor)
        # trading_days x n_factors x n_stocks. Ogni giorno so quanto investire su ogni compagnia all'interno di ognuno dei fattori
        Q[i, :, :] = money_on_stock(period, eigenvectors)
        # ritorni dei fattori di rischio per ogni periodo
        factors = risk_factors(period, Q[i], eigenvectors)
        # Ottenuti i fattori di rischio si procede con la stima del processo dei residui per ogni compagnia.
        for stock in df_returns.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            beta0, betas, conf_inter, residuals, pred, _ = regression(
                factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])
            # Avoid estimation of non stationary residuals
            p_value = adfuller(residuals)[1]
            if p_value > 0.05:
                pass
            else:
                alpha_values[i, stock_idx] = beta0 * lookback_for_factors
                beta_tensor[i, stock_idx, :] = betas
                res[i, stock_idx, :] = residuals
                X = np.cumsum(residuals)
                dis_res[i, stock_idx, :] = X

                if method == 'constant_speed':
                    X = np.append(X, X[-1])
                    parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(
                        X)
                    a, b = parameters[0], parameters[1]
                    # Registro la dinamica di b nel tempo, memorizzando media e intervallo di confidenza. L'idea successiva è quella di modellizzare b
                    # tramite un GAS model.
                    # discrete_conf_int contains both the conf_int for a (0) and for b (1)
                    b_values[i, stock_idx, 0], b_values[i,
                                                        stock_idx, 1:] = b, discrete_conf_int[1, :]
                    R_squared[i, stock_idx] = r2_score(
                        X[:-1], np.array(discrete_pred))
                    # Registro tutti i residui per i test sulla normalità ed "indipendenza" da fare successivamente
                    dis_res_reg[i, stock_idx, :] = discrete_resid

                if method == 'gas_modelization':
                    if targeting_estimation:
                        init_params = np.random.uniform(0, 1, size=3)
                        b, a, xi, est, std = estimation(
                            loglikelihood_after_targ, X, init_params, targeting_estimation=targeting_estimation)
                    else:
                        # est -> omega, a, alpha, beta
                        init_params = np.random.uniform(0, 1, size=4)
                        # init_params = initial_value(loglikelihood, 100, X, 4)
                        b, a, xi, est = estimation(
                            loglikelihood, X, init_params)
                    estimates[i, stock_idx, :] = est.x
                    b_values_gas[i, stock_idx, :] = b
                    a_values_gas[i, stock_idx, :] = a
                    b = b[-1]  # Mi serve solo l'ultimo valore per lo score
                    if b < 0:  # se b è negativo, sostituiscilo con quello stimato supponendo sia costante nella finestra temporale
                        X = np.append(X, X[-1])
                        parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(
                            X)
                        a, b = parameters[0], parameters[1]
                        c += 1

                if b == 0.0:
                    print(f'B NULLO PER {stock}')
                    break

                k = -np.log(b) * lookback_for_factors
                if k < lookback_for_factors / (0.5 * lookback_for_residual):
                    score[i, stock_idx] = 0
                    cc += 1
                else:
                    m = a / (1 - b)
                    if method == 'constant_speed':
                        sgm_eq[i, stock_idx] = np.std(
                            discrete_resid) * np.sqrt(1 / (1 - b * b))
                    if method == 'gas_modelization':
                        sgm_eq[i, stock_idx] = np.std(
                            xi) * np.sqrt(1 / (1 - b * b))
                    # naive method. Keep in mind that s-score depends on the risk factors
                    score[i, stock_idx] = -m / sgm_eq[i, stock_idx] - \
                        beta0 / (k * sgm_eq[i, stock_idx])

    with open(go_up(1) + f'/saved_data/negative_b_{os.getpid()}', 'w', encoding='utf-8') as file:
        file.write(
            f'Number of negative b values for process {os.getpid()}: {c} \n')
        file.write(
            f'Number of stock w mean reversion refused {os.getpid()}: {cc}')

    df_score = pd.DataFrame(score, columns=df_returns.columns)
    if export:
        if method == 'gas_modelization':
            # df_score.to_csv(go_up(1) + f'/saved_data/df_score_gas_{os.getpid()}.csv', index=False)
            # np.save(go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
            # np.save(go_up(1) + f'/saved_data/alpha_values_{os.getpid()}', alpha_values)
            # np.save(go_up(1) + f'/saved_data/sgm_eq_gas_{os.getpid()}', sgm_eq)
            # np.save(go_up(1) + f'/saved_data/b_gas_{os.getpid()}', b_values_gas)
            # np.save(go_up(1) + f'/saved_data/a_gas_{os.getpid()}', a_values_gas)
            # np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)
            np.save(go_up(1) + f'/saved_data/dis_res_{os.getpid()}', dis_res)
            np.save(
                go_up(1) + f'/saved_data/estimates_{os.getpid()}', estimates)
            # np.save(go_up(1) + f'/saved_data/res_{os.getpid()}', res)

        if method == 'constant_speed':
            # df_score.to_csv(go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)
            # np.save(go_up(1) + f'/saved_data/dis_res_reg_{os.getpid()}', dis_res_reg)
            # np.save(go_up(1) + f'/saved_data/b_values_{os.getpid()}', b_values)
            # np.save(go_up(1) + f'/saved_data/R_squared_{os.getpid()}', R_squared)
            np.save(
                go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
            # np.save(go_up(1) + f'/saved_data/alpha_values_{os.getpid()}', alpha_values)
            # np.save(go_up(1) + f'/saved_data/sgm_eq_{os.getpid()}', sgm_eq)
            np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)
            # np.save(go_up(1) + f'/saved_data/dis_res_{os.getpid()}', dis_res)


def only_scoring(dis_res, df_returns, method, lookback_for_factors, lookback_for_residual, targeting_estimation=False):
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    score = np.zeros(shape=(trading_days, n_stocks))
    if targeting_estimation:
        num_par = 3
    else:
        num_par = 4
    estimates = np.zeros(shape=(trading_days, n_stocks, num_par))
    c = 0
    cc = 0
    ccc = 0

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days), desc=f'{os.getpid()}'):
        for stock in df_returns.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            X = dis_res[i, stock_idx, :]
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

    #             b = b[-1]
    #             if b < 0:
    #                 X = np.append(X, X[-1])
    #                 parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(X)
    #                 a, b = parameters[0], parameters[1]
    #                 c += 1
    #         if b == 0.0:
    #             print(f'B NULLO PER {stock}')
    #             break
    #
    #         k = -np.log(b) * lookback_for_factors
    #         if k < lookback_for_factors / (0.5 * lookback_for_residual):
    #             score[i, stock_idx] = 0
    #             cc += 1
    #         else:
    #             m = a / (1 - b)
    #             sgm_eq = np.std(xi) * np.sqrt(1 / (1 - b**2))
    #             score[i, stock_idx] = -m / sgm_eq
    #
    # with open(go_up(1) + f'/saved_data/negative_b_{os.getpid()}', 'w', encoding='utf-8') as file:
    #     file.write(f'Number of negative b values for process {os.getpid()}: {c} \n')
    #     file.write(f'Number of stock w mean reversion refused {os.getpid()}: {cc}')
    #     file.write(f'Nan scores {os.getpid()}: {ccc}')
    #
    # df_score = pd.DataFrame(score, columns=df_returns.columns)
    if method == 'gas_modelization':
        # df_score.to_csv(go_up(1) + f'/saved_data/df_score_gas_{os.getpid()}.csv', index=False)
        np.save(go_up(1) + f'/saved_data/estimates_{os.getpid()}', estimates)
    if method == 'constant_speed':
        df_score.to_csv(
            go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)


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
    parser.add_argument("-vn", "--variable_number", action='store_true',
                        help=("Use a variable number of PCA components. Each time the explained variance is 0.55. The default is False"))
    parser.add_argument('-n', '--n_components', type=int, default=15,
                        help='Number of PCA components to keep. The default is 15.')
    parser.add_argument("-g", "--gas", action='store_true',
                        help=("Use gas estimation for the mean reverting speed. The default is False."))
    parser.add_argument("-sp", "--spy", action='store_true',
                        help=("Scoring for SPY."))
    parser.add_argument('-r', '--range', action='store_true',
                        help='Select a specific time range between 1995-01-03 to 2020-12-31')
    parser.add_argument('-s', '--save_outputs', action='store_false',
                        help='Choose whether or not to save the outputs. The default is True')
    parser.add_argument("-t", "--targ_est", action='store_true')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    start = time.time()

    # df_returns = pd.read_csv(go_up(1) +
    #                          "/saved_data/ReturnsData.csv").drop(columns=['HFC', 'MXIM', 'XLNX', 'KSU', 'RRD'])[:4030]
    df_returns = pd.read_csv(go_up(1) +
                             "/saved_data/ReturnsData.csv")[:4030]
    dis_res = np.load(go_up(1) + "/saved_data/dis_res60.npy")

    if args.spy:
        spy = pd.read_csv(go_up(1) + "/saved_data/spy.csv")
        SPY_beta(df_returns, spy)

    # df = [df_returns[:1510], df_returns[1258:2769], df_returns[2517:4028], df_returns[3776:5287], df_returns[5035:]]
    # df = [df_returns[:1000], df_returns[748:1749], df_returns[1497:2498], df_returns[2246:3247], df_returns[2995:3996], df_returns[3744:4745], df_returns[4493:5494],df_returns[5242:]]
    df = [df_returns[:750], df_returns[498:1249], df_returns[997:1748], df_returns[1496:2247],
          df_returns[1995:2746], df_returns[2494:3245], df_returns[2993:3744], df_returns[3492:]]
    dr = [dis_res[:750, :, :], dis_res[498:1249, :, :], dis_res[997:1748, :, :], dis_res[1496:2247, :, :],
          dis_res[1995:2746, :, :], dis_res[2494:3245, :, :], dis_res[2993:3744, :, :], dis_res[3492:, :, :]]

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
        generate_data(df_returns, args.n_components, method)
        end = time.time()

    else:
        for iter in range(1):
            np.random.seed()
            processes = [mp.Process(target=generate_data, args=(i, args.n_components, method, args.targ_est, 252, 60, args.save_outputs)) for i in df]
            # processes = [mp.Process(target=only_scoring, args=(
            #     i, j, method, 252, 60)) for i, j in zip(dr, df)]
            os.system('rm tmp/*')
            for p in processes:
                p.start()
                time.sleep(0.5)
            for p in processes:
                p.join()
            end = time.time()

            telegram_send.send(messages=[f'Iteration {iter} terminated'])
            pidnums = [int(x) for x in os.listdir('tmp')]
            pidnums.sort()
            if args.gas:
                # file_list = ['df_score_gas', 'beta_tensor', 'alpha_values', 'Q', 'b_gas', 'a_gas', 'dis_res', 'res', 'sgm_eq_gas']
                file_list = ['df_score_gas', 'estimates']
            else:
                # file_list = ['beta_tensor', 'Q', 'dis_res', 'df_score', 'dis_res_reg', 'b_values', 'R_squared', 'sgm_eq']
                file_list = ['df_score', 'beta_tensor', 'Q']
            logging.info('Merging files...')
            file_name = 'ScoreData_60volint'
            file_merge(pidnums, file_list, file_name)
            logging.info('Removing splitted files...')
            remove_file(pidnums, file_list)
            os.system('rm tmp/*')

    time_elapsed = (end - start)
    logging.info('Time required for generate s-scores: %.2f seconds' %
                 time_elapsed)
