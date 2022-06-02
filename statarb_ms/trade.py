import argparse
import itertools
import logging
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import telegram_send
from makedir import go_up
from post_processing import file_merge, sharpe_ratio
from tqdm import tqdm
from factors import pca, money_on_stock


def trading(df_returns, df_score, Q, beta_tensor, lookback_for_residual, epsilon=0.0005):
    '''This function run a back test of the replication of the statistical arbitrage strategy by Avellaneda Lee (Avellaneda and Lee, 2010. DOI: 10.1080/14697680903124632).

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns.
    df_score : pandas.core.frame.DataFrame
        Dataframe of stock prices.
    Q : numpy.ndarray
        Numpy array of shape (n_factors, n_stocks). The i-j element indicates the amount of money invested in
        the company j in the factor i.
    beta_tensor : numpy.ndarray
        Numpy array of shape (trading_days, n_factors, n_stocks). For each day it contains the projections of one stock on the factors.
    epsilon : float
        Slippage.
    s_bo, s_so, s_bc, s_sc : float
        Thresholds for (respectively) buy to open, short to open, close long, close short.

    Returns
    -------
    PnL : numpy.ndarray
        Profit and Loss of the strategy.
    '''

    PnL = np.zeros(df_score.shape[0])
    PnL[0] = 1.00
    fraction = 0.01
    lookback_for_factors = 252
    daily_PnL = np.zeros(shape=df_returns.shape)
    state = np.array(['c' for i in range(df_returns.shape[1])])
    day_counter_long = np.zeros(shape=df_returns.shape[1], dtype=int)
    day_counter_short = np.zeros(shape=df_returns.shape[1], dtype=int)
    perc_positions = np.zeros(shape=(df_returns.shape[0], 3))
    invest_amount = np.zeros(shape=(df_returns.shape[0] + 1))

    prod = np.zeros(shape=((df_score.shape[0] - 1) * df_score.shape[1]))
    K = np.zeros(shape=((df_score.shape[0] - 1) * df_score.shape[1]))
    fact_cont = np.zeros(shape=df_score.shape[0] - 1)
    stock_cont = np.zeros(shape=df_score.shape[0] - 1)
    fees = np.zeros(shape=df_score.shape[0] - 1)


    if lookback_for_residual == 50:
        s_bo, s_so, s_bc, s_sc = 1.25, 1.25, 0.75, 0.5
    elif lookback_for_residual == 60:
        s_bo, s_so, s_bc, s_sc = 1.10, 1.15, 0.80, 0.70
    elif lookback_for_residual == 70:
        s_bo, s_so, s_bc, s_sc = 1.10, 1.20, 0.80, 0.75
    elif lookback_for_residual == 80:
        s_bo, s_so, s_bc, s_sc = 1.15, 1.10, 0.75, 0.55
    elif lookback_for_residual == 90:
        s_bo, s_so, s_bc, s_sc = 1.25, 1.25, 0.75, 0.5
    elif lookback_for_residual == 100:
        s_bo, s_so, s_bc, s_sc = 1.25, 1.25, 0.75, 0.5


    for day in tqdm(range(df_score.shape[0] - 1)):
        counter_no_trades = 0
        factor_contribution = []
        stock_contribution = []
        for stock in df_score.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            if df_score[stock][day] == 0:
                counter_no_trades += 1
                continue
            if df_score[stock][day] < -s_bo and (state[stock_idx] == 'c'):
                state[stock_idx] = 'l'
                k = PnL[day] * fraction / (1 + (np.dot(beta_tensor[day, stock_idx, :], Q[day,:,stock_idx])))
                daily_PnL[day, stock_idx] = k * (df_returns[stock][day + lookback_for_factors] - np.matmul(beta_tensor[day, stock_idx, :], np.matmul(Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                invest_amount[day+1] = np.dot(beta_tensor[day, stock_idx, :], Q[day,:,stock_idx])
                continue
            if (day > 0) and (df_score[stock][day] < -s_sc) and (state[stock_idx] == 'l'):
                day_counter_long[stock_idx] += 1
                k = PnL[day - day_counter_long[stock_idx]] * fraction / (1 + (np.dot(beta_tensor[day - day_counter_long[stock_idx], stock_idx, :], Q[day - day_counter_long[stock_idx],:,stock_idx])))
                daily_PnL[day, stock_idx] = k * (df_returns[stock][day + lookback_for_factors] - np.matmul(beta_tensor[day - day_counter_long[stock_idx], stock_idx, :], np.matmul(Q[day - day_counter_long[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                continue
            if df_score[stock][day] > s_so and (state[stock_idx] == 'c'):
                state[stock_idx] = 's'
                k = PnL[day] * fraction / (1 + np.dot(beta_tensor[day, stock_idx, :], Q[day,:,stock_idx]))
                daily_PnL[day, stock_idx] = k * (-df_returns[stock][day + lookback_for_factors] + np.matmul(beta_tensor[day, stock_idx, :], np.matmul(Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                invest_amount[day+1] = np.dot(beta_tensor[day, stock_idx, :], Q[day,:,stock_idx])
                continue
            if (day > 0) and (df_score[stock][day] > s_bc) and (state[stock_idx] == 's'):
                day_counter_short[stock_idx] += 1
                k = PnL[day - day_counter_short[stock_idx]] * fraction / (1 + np.dot(beta_tensor[day - day_counter_short[stock_idx], stock_idx, :], Q[day - day_counter_short[stock_idx],:,stock_idx]))
                daily_PnL[day, stock_idx] = k * (-df_returns[stock][day + lookback_for_factors] + np.matmul(beta_tensor[day - day_counter_short[stock_idx], stock_idx, :], np.matmul(Q[day - day_counter_short[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                continue
            if (day > 0) and (df_score[stock][day] > -s_sc) and (state[stock_idx] == 'l'):
                day_counter_long[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue
            if (day > 0) and (df_score[stock][day] < s_bc) and (state[stock_idx] == 's'):
                day_counter_short[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue
            else:
                counter_no_trades += 1
                continue
        perc_positions[day,0] = np.count_nonzero(state == 'l')/df_score.shape[1]
        perc_positions[day,1] = np.count_nonzero(state == 's')/df_score.shape[1]
        perc_positions[day,2] = np.count_nonzero(state == 'c')/df_score.shape[1]
        PnL[day + 1] = PnL[day] + daily_PnL[day,:].sum() - (np.abs((invest_amount[day+1] - invest_amount[day]))).sum() * epsilon

    return PnL, perc_positions, fact_cont, stock_cont, fees


def spy_trading(df_returns, Q):

    df_returns = np.array(df_returns[252:])
    trading_days = df_returns.shape[0]
    n_stocks = df_returns.shape[1]

    A = 1 / Q[0,0,:].sum()
    Q = A*Q[0,0,:] # normalized to one, that is the amount of money at time zero
    daily_PnL = np.ones(shape=trading_days)

    for i in tqdm(range(trading_days - 1)):
        daily_PnL[i+1] = daily_PnL[i] + (Q*df_returns[i+1]).sum()

    return daily_PnL


def grid_search(hyperparameters, score_data):

    results = pd.DataFrame(index=range(len(hyperparameters)), columns=[
                           'Hyperparameters', 'SharpeRatio'])

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for hyper in hyperparameters:
        idx = hyperparameters.index(hyper)
        s_bo, s_so, s_bc, s_sc = hyper
        pnl, perc_positions, fact_cont, stock_cont, fees = trading(
            df_returns, df_score, Q, beta_tensor, s_bo=hyper[0], s_so=hyper[1], s_bc=hyper[2], s_sc=hyper[3])
        spy = np.load('/mnt/saved_data/PnL/pnl_firstcomp.npy')[:pnl.shape[0]]
        s_ratio = sharpe_ratio(pnl, spy, period=pnl.shape[0])[0]
        # print(s_ratio[0], type(s_ratio))
        results['Hyperparameters'][idx] = hyper
        results['SharpeRatio'][idx] = s_ratio
    results.to_pickle(f'/mnt/saved_data/gridsearch_{os.getpid()}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('-s', '--spy', action='store_true',
                        help='Generate PnL for the Buy and Hold strategy of the first principal component')
    parser.add_argument('-vi', '--vol_int', action='store_true', help='Use return dataframe weighted  with volume information')
    parser.add_argument('-g', '--grid_search', action='store_true',
                        help='Grid search for model hyperparameters selection')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    start = time.time()

    if args.vol_int:
        # Le quantità Q, beta_tensor e df_score sono state generate tramite il dataframe pesato con i volumi. Il primo valore di tale dataframe utilizza le informazioni dei ritorni semplici indicizzati da 9 e le informazioni dei volumi relativi al 10° giorno. Il 9° valore del dataframe dei ritorni corrisponde alla quantità (R(10) - R(9)) / R(9), quindi al ritorno che leggo a chiusura del 10° giorno. A tale chiusura leggo anche il valore del volume al 10° giorno.

        # Di conseguenza, il df_returns qui di seguito deve partire dal 9° valore. Q, beta_tensor e df_score devono finire invece al -10, siccome sono stati generati dal dataframe pesato coi volumi che in realtà parte dalle infor a chiusura del 10° giorno e sfora sempre di 10 giorni la fine di df_returns.

        logging.info('I am using scores obtained from modified returns')
        time.sleep(0.3)
        df_returns = pd.read_pickle("/mnt/saved_data/returns/ReturnsData.pkl")[9:]
        df_returns.index = range(df_returns.shape[0]) # This dataframe has shape ReturnsData.shape[0] - 10
        Q = np.load('/mnt/saved_data/Qs/Q_volint.npy')[:-10,:,:]
        length = int(input('Lenght of estimation window for AR(1) parameters: '))
        name = input('Name of the s-score data file: ')
        beta_tensor = np.load(f'/mnt/saved_data/beta_tensor{length}_volint.npy')[:-10,:,:]

        df_score = pd.read_pickle(f'/mnt/saved_data/scores/{name}.pkl')[:-10]
        df_score.index = range(df_score.shape[0])

    else:
        logging.info('I am using scores obtained from simple returns')
        time.sleep(0.3)
        df_returns = pd.read_pickle("/home/danielemdn/Documents/saved_data/returns/ReturnsData.pkl")[:4030]
        Q = np.load('/home/danielemdn/Documents/saved_data/Qs/Q.npy')
        length = int(input('Lenght of estimation window for AR(1) parameters: '))
        name = input('Name of the s-score data file: ')
        beta_tensor = np.load(f'/home/danielemdn/Documents/saved_data/betas/beta_tensor{length}.npy')
        df_score = pd.read_pickle(f'/home/danielemdn/Documents/saved_data/scores/{name}.pkl')

    if args.grid_search:
        close_range = np.arange(0.5, 0.85, 0.05)
        open_range = np.arange(1.1, 1.35, 0.05)
        closes = list(itertools.permutations(close_range, 2))
        opens = list(itertools.permutations(open_range, 2))

        hyperparameters = []
        for o in opens:
            for c in closes:
                hyperparameters.append([o[0], o[1], c[0], c[1]])
        hyperparameters = [hyperparameters[i: i + 63]
                           for i in range(0, len(hyperparameters), 63)]

        processes = [mp.Process(target=grid_search, args=(
            hyperparam, df_score)) for hyperparam in hyperparameters]

        os.system('rm tmp/*')

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        end = time.time()

        telegram_send.send(messages=[f'GridSearch on {name} terminated'])
        pidnums = [int(x) for x in os.listdir('tmp')]
        pidnums.sort()
        logging.info('Merging files, then remove the splitted ones...')
        file_merge(pidnums, ['gridsearch'])
        os.system('rm tmp/*')

        gs = pd.read_pickle(f'/mnt/saved_data/gridsearch_{name}.pkl')
        idxs = np.where(gs.values[:,1] == gs['SharpeRatio'].max())
        gs = gs.iloc[idxs]
        gs.to_pickle(f'/mnt/saved_data/gridsearch_{name}.pkl')
        exit()

    if args.spy:
        logging.info('Starting spy trading... ')
        time.sleep(1)
        spy_pnl = spy_trading(df_returns, Q)
        name = input('Name of the file that will be saved (spy): ')
        np.save(f'/mnt/saved_data/PnL/{name}', spy_pnl)

    else:
        for i in range(1):
            pnl, perc_positions, fact_cont, stock_cont, fees = trading(
                df_returns, df_score, Q, beta_tensor, lookback_for_residual=length)
            name = input('Name of the file that will be saved (strategy): ')
            np.save(f'/home/danielemdn/Documents/saved_data/PnL/{name}', pnl)
        # name = input('Name of the file that will be saved (positions percentage): ')
        # np.save(go_up(1) + f'/saved_data/{name}', perc_positions)

    end = time.time()
    time_elapsed = (end - start) / 60
    logging.info('Elapsed time: %.2f minutes' % time_elapsed)
