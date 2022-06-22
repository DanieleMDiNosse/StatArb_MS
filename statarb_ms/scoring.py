# import numpy as np
import argparse
import logging
import multiprocessing as mp
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import telegram_send
from data import price_data
from factors import money_on_stock, pca, risk_factors
from gas import estimation
from loglikelihood import loglikelihood
from post_processing import LM_test_statistic, file_merge
from regression_parameters import auto_regression, regression
from scipy.stats import normaltest
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
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

    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    alphas = np.zeros(shape=(trading_days, n_stocks))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days), desc=f'{os.getpid()}'):
        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        # [0,252[, [1,253[ ecc -> ogni period comprende un anno di trading (252 giorni)
        period = df_returns[i:lookback_for_factors + i]
        eigenvalues, eigenvectors, exp_var = pca(period, n_components=n_factor)
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
                continue

            alphas[i, stock_idx] = beta0 * lookback_for_factors
            beta_tensor[i, stock_idx, :] = betas
            res[i, stock_idx, :] = residuals
            X = np.cumsum(residuals)
            dis_res[i, stock_idx, :] = X

    if export:
        np.save(
            f'/mnt/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
        np.save(f'/mnt/saved_data/alphas_{os.getpid()}', alphas)
        np.save(f'/mnt/saved_data/Q_{os.getpid()}', Q)
        np.save(f'/mnt/saved_data/dis_res_{os.getpid()}', dis_res)
        np.save(f'/mnt/saved_data/res_{os.getpid()}', res)


def only_scoring(dis_res, df_returns, method, link_fun, n_iter, lookback_for_factors, lookback_for_residual, targeting_estimation=False):

    if targeting_estimation:
        num_par = 3
    else:
        num_par = 4

    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    score = np.zeros(shape=(trading_days, n_stocks))
    estimates = np.zeros(shape=(trading_days, n_stocks, num_par + 1))
    bs = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    r2 = np.zeros(shape=(trading_days, n_stocks))
    const_AR_par = np.zeros(shape=(trading_days, n_stocks, 2))
    AR_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    AR_res_gas = np.zeros(
        shape=(trading_days, n_stocks, lookback_for_residual))
    sgm_eq = np.zeros(shape=(trading_days, n_stocks))
    kappas = np.zeros(shape=(trading_days, n_stocks))

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

                r2[i, stock_idx] = r2_score(X[:-1], np.array(discrete_pred))
                const_AR_par[i, stock_idx, :] = parameters
                AR_res[i, stock_idx, :] = xi

            if method == 'gas_modelization':
                if targeting_estimation:
                    b, a, xi, est, std = estimation(
                        X, n_iter, targeting_estimation=targeting_estimation)
                else:
                    b, a, xi, est = estimation(X, n_iter, link_fun)
                AR_res_gas[i, stock_idx, :] = xi
                estimates[i, stock_idx, :-1] = est
                estimates[i, stock_idx, -1] = 1  # sigma = 1 by assumption
                bs[i, stock_idx, :] = b
                b = b[-1]

                if b < 0:
                    X = np.append(X, X[-1])
                    parameters, discrete_pred, xi, discrete_conf_int = auto_regression(
                        X)
                    a, b = parameters[0], parameters[1]
                    c += 1

            if b == 0.0:
                cc += 1
                continue
            else:
                k = -np.log(b) * lookback_for_factors
                kappas[i, stock_idx] = k
                if k < lookback_for_factors / (0.5 * lookback_for_residual):
                    score[i, stock_idx] = 0
                    ccc += 1
                else:
                    m = a / (1 - b)
                    sgm_eq[i, stock_idx] = np.std(xi) * np.sqrt(1 / (1 - b**2))
                    score[i, stock_idx] = -m / sgm_eq[i, stock_idx]
    try:
        telegram_send.send(messages=[f'Total number of estimation for process {os.getpid()}: {n_stocks * trading_days}', f'Number of negative b values for process {os.getpid()}: {c}',
                           f'Number of stock with speed of mean reversion refused for process {os.getpid()}: {ccc}', f'Number of zero b for process {os.getpid()}: {cc}'])
    except Exception:
        logging.warning(
            'Unable to send info to Telegram bot, probably due to no internet connection')
        messages = [f'Total number of estimation for process {os.getpid()}: {n_stocks * trading_days}', f'Number of negative b values for process {os.getpid()}: {c}',
                    f'Number of stock with speed of mean reversion refused for process {os.getpid()}: {ccc}', f'Number of zero b for process {os.getpid()}: {cc}']
        logging.info(messages)

    df_score = pd.DataFrame(score, columns=df_returns.columns)

    if method == 'gas_modelization':
        logging.info('Saving files...')
        df_score.to_pickle(f'/mnt/saved_data/df_score_gas_{os.getpid()}.pkl')
        np.save(f'/mnt/saved_data/estimates_gas_{os.getpid()}', estimates)
        np.save(f'/mnt/saved_data/bs_{os.getpid()}', bs)
        np.save(f'/mnt/saved_data/sgm_eq_gas_{os.getpid()}', sgm_eq)
        np.save(f'/mnt/saved_data/kappas_gas_{os.getpid()}', kappas)
        np.save(f'/mnt/saved_data/AR_res_gas_{os.getpid()}', AR_res_gas)

    if method == 'constant_speed':
        logging.info('Saving files...')
        df_score.to_pickle(f'/mnt/saved_data/df_score_{os.getpid()}.pkl')
        np.save(f'/mnt/saved_data/r2_{os.getpid()}', r2)
        np.save(f'/mnt/saved_data/const_AR_par_{os.getpid()}', const_AR_par)
        np.save(f'/mnt/saved_data/sgm_eq_{os.getpid()}', sgm_eq)
        np.save(f'/mnt/saved_data/AR_res_{os.getpid()}', AR_res)
        np.save(f'/mnt/saved_data/kappas_{os.getpid()}', kappas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-s", "--scoring", action='store_true',
                        help='If passed, compute only the s-scores, provided residuals are already been computed.')
    parser.add_argument("-g", "--gas", action='store_true',
                        help=("Use gas estimation for the mean reverting speed. The default is False."))
    parser.add_argument('-lf', '--link_fun', type=int, help='In GAS filtering, choose what type of link function/distribution is adopted. 0 -> Identity link function with N(0,1) distribution; 1-> Logistic link function with N(0,1) distribution; 2-> Identity link function with Student t distribution.')
    parser.add_argument('-vi', '--vol_int', action='store_true',
                        help='Use return dataframe weighted  with volume information')
    parser.add_argument('-t', '--test_set', action='store_true',
                        help='Use test set with the hyperparameters tuned on the validation set')
    parser.add_argument("-nit", "--n_iter", type=int,
                        help='Number of Nelder-Mead optimization (minus 1) before the final BFGS optimization.', default=2)

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    plt.style.use('seaborn')
    warnings.filterwarnings('ignore')
    start = time.time()
    np.random.seed()

    if args.gas == True:
        method = 'gas_modelization'

        if args.link_fun == 0:
            link_fun = 'identity'
        if args.link_fun == 1:
            link_fun = 'logistic'
        if args.link_fun == 2:
            link_fun = 'identity_student'

    else:
        method = 'constant_speed'

    if args.vol_int:
        logging.info(
            'I am computing scores from the volume integrated returns')

        if args.test_set:
            logging.info('Using test set')
            df_returns = pd.read_pickle(
                "/mnt/saved_data/returns/ReturnsVolData.pkl")[4029:]
        else:
            df_returns = pd.read_pickle(
                "/mnt/saved_data/returns/ReturnsVolData.pkl")[:4030]
            # stocks = np.random.randint(0, df_returns.shape[1], size=10)
            # print(f'{df_returns.columns[stocks]}')
            # df_returns = df_returns[df_returns.columns[stocks]]

    else:
        logging.info('I am computing scores from the simple returns')

        if args.test_set:
            logging.info('Using test set')
            df_returns = pd.read_pickle(
                "/mnt/saved_data/returns/ReturnsData.pkl")[4029:]
        else:
            df_returns = pd.read_pickle(
                "/mnt/saved_data/returns/ReturnsData.pkl")[:4030]

    if args.test_set:
        df = [df_returns[:565], df_returns[313:879], df_returns[627:1193], df_returns[941:1507],
              df_returns[1255:1821], df_returns[1569:2135], df_returns[1883:2449], df_returns[2197:]]
    else:
        df = [df_returns[:750], df_returns[498:1249], df_returns[997:1748], df_returns[1496:2247],
              df_returns[1995:2746], df_returns[2494:3245], df_returns[2993:3744], df_returns[3492:]]

    if args.scoring:
        logging.info('Computing the s-scores...')
        time.sleep(0.3)
        length = int(input('Length of the estimation window for the scores: '))
        name = input('Name of the dis_res file: ')
        logging.info(f'{name}, {length}')
        dis_res = np.load(f"/mnt/saved_data/dis_res/{name}.npy")#[:, stocks, :]

        try:
            telegram_send.send(
                messages=[f'========== Start Scoring, Length: {length}, {time.asctime()} =========='])
        except Exception:
            logging.warning(
                'Unable to send infos to Telegram bot, probably due to no internet connection')

        if args.test_set:
            dr = [dis_res[:565, :, :], dis_res[313:879, :, :], dis_res[627:1193, :, :], dis_res[941:1507, :, :],
                  dis_res[1255:1821, :, :], dis_res[1569:2135, :, :], dis_res[1883:2449, :, :], dis_res[2197:, :, :]]
        else:
            dr = [dis_res[:750, :, :], dis_res[498:1249, :, :], dis_res[997:1748, :, :], dis_res[1496:2247, :, :],
                  dis_res[1995:2746, :, :], dis_res[2494:3245, :, :], dis_res[2993:3744, :, :], dis_res[3492:, :, :]]

        processes = [mp.Process(target=only_scoring, args=(
            i, j, method, link_fun, args.n_iter, 252, length)) for i, j in zip(dr, df)]

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
            file_list = ['df_score_gas', 'estimates_gas',
                         'bs', 'sgm_eq_gas', 'kappas_gas', 'AR_res_gas']
        else:
            file_list = ['kappas', 'AR_res', 'sgm_eq',
                         'const_AR_par', 'r2', 'df_score']

        logging.info('Merging files, then remove splitted ones...')
        file_merge(pidnums, file_list)
        os.system('rm tmp/*')

        try:
            telegram_send.send(
                messages=[f'============= End Scoring for {length} =============='])
        except Exception:
            print(
                'Unable to send infos to Telegram bot, probably due to no internet connection')

    else:
        logging.info('Estimating residual process...')
        time.sleep(0.3)
        length = int(
            input('Lenght of the estimation window for the residuals: '))
        processes = [mp.Process(target=generate_data, args=(
            i, 15, 252, length, True)) for i in df]

        try:
            telegram_send.send(
                messages=[f'==========  Start Residuals, Length: {length}, {time.asctime()} =========='])
        except Exception:
            logging.warning(
                'Unable to send infos to Telegram bot, probably due to no internet connection')

        os.system('rm tmp/*')
        for p in processes:
            p.start()
            time.sleep(0.5)
        for p in processes:
            p.join()
        end = time.time()

        pidnums = [int(x) for x in os.listdir('tmp')]
        pidnums.sort()
        file_list = ['res', 'dis_res', 'beta_tensor', 'alphas', 'Q']

        logging.info('Merging files, then remove splitted ones...')
        file_merge(pidnums, file_list)
        os.system('rm tmp/*')

        try:
            telegram_send.send(
                messages=[f'============= End Residuals for {length} =============='])
        except Exception:
            print(
                'Unable to send infos to Telegram bot, probably due to no internet connection')

    time_elapsed = (end - start)
    logging.info('Time required for generate s-scores: %.2f seconds' %
                 time_elapsed)
