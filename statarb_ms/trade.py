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
from post_processing import file_merge, remove_file, sharpe_ratio
from tqdm import tqdm


def trading(df_returns, df_score, Q, beta_tensor, epsilon=0.0005, s_bo=1.25, s_so=1.25, s_bc=0.75, s_sc=0.5):
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
    fraction = 0.02
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

    for day in tqdm(range(df_score.shape[0] - 1)):
        counter_no_trades = 0
        factor_contribution = []
        stock_contribution = []
        for stock in df_score.columns:
            stock_idx = df_returns.columns.get_loc(stock)

            if (df_score[stock][day] == 0):
                counter_no_trades += 1
                continue
# Beta tensor e Q hanno dei valori massimi che sono di gran lunga maggiori rispetto a quelli dei vecchi beta tensor e Q. Vedi un po' di capire perché sono valutati in questo modo
# Forse il problema sta proprio qui, sebbene abbia controllato più e più volte e sembra che tutto sia fatto bene, almeno nella funzione generate_data.

# Grafica i contributi dei fattori e delle stock giorno per giorno. L'idea è di capire cosa causa gli spike nel PnL fra i due. Grafica anche i costi di transazione.
            if (df_score[stock][day] < -s_bo) and (state[stock_idx] == 'c'):
                state[stock_idx] = 'l'

                if (beta_tensor[day, stock_idx, :].any(axis=0) == 0) or (Q[day, :, stock_idx].any(axis=0) == 0):
                    daily_PnL[day, stock_idx] = 0
                    invest_amount[day + 1] = 0
                else:
                    prod[day * stock_idx] = np.dot(beta_tensor[day,
                                                   stock_idx, :], Q[day, :, stock_idx])
                    k = PnL[day] * fraction / (1 + prod[day * stock_idx])

                    factor_contribution.append(k * np.matmul(beta_tensor[day, stock_idx, :], np.matmul(
                        Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                    stock_contribution.append(k * df_returns[stock][day + lookback_for_factors])

                    daily_PnL[day, stock_idx] = k * \
                        (df_returns[stock][day +
                         lookback_for_factors] - np.matmul(beta_tensor[day, stock_idx, :], np.matmul(
                             Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                    K[day * stock_idx] = k
                    invest_amount[day + 1] = prod[day * stock_idx]
                continue

            if (day > 0) and (df_score[stock][day] < -s_sc) and (state[stock_idx] == 'l'):
                day_counter_long[stock_idx] += 1

                if (beta_tensor[day - day_counter_long[stock_idx], stock_idx, :].any(axis=0) == 0) or (Q[day - day_counter_long[stock_idx], :, stock_idx].any(axis=0) == 0):
                    daily_PnL[day, stock_idx] = 0
                else:
                    prod[day * stock_idx] = np.dot(beta_tensor[day - day_counter_long[stock_idx],
                                                   stock_idx, :], Q[day - day_counter_long[stock_idx], :, stock_idx])
                    k = PnL[day - day_counter_long[stock_idx]] * \
                        fraction / (1 + prod[day * stock_idx])

                    factor_contribution.append(k * np.matmul(beta_tensor[day - day_counter_long[stock_idx], stock_idx, :], np.matmul(
                        Q[day - day_counter_long[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                    stock_contribution.append(k * df_returns[stock][day + lookback_for_factors])

                    daily_PnL[day, stock_idx] = k * \
                        (df_returns[stock][day +
                         lookback_for_factors] - np.matmul(beta_tensor[day - day_counter_long[stock_idx], stock_idx, :], np.matmul(
                             Q[day - day_counter_long[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                    K[day * stock_idx] = k
                continue

            if df_score[stock][day] > s_so and (state[stock_idx] == 'c'):
                state[stock_idx] = 's'

                if (beta_tensor[day, stock_idx, :].any(axis=0) == 0) or (Q[day, :, stock_idx].any(axis=0) == 0):
                    daily_PnL[day, stock_idx] = 0
                    invest_amount[day + 1] = 0
                else:
                    prod[day * stock_idx] = np.dot(beta_tensor[day,
                                                   stock_idx, :], Q[day, :, stock_idx])
                    k = PnL[day] * fraction / (1 + prod[day * stock_idx])

                    factor_contribution.append(k * np.matmul(beta_tensor[day, stock_idx, :], np.matmul(
                        Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                    stock_contribution.append(k * df_returns[stock][day + lookback_for_factors])

                    daily_PnL[day, stock_idx] = k * \
                        (-df_returns[stock][day +
                         lookback_for_factors] + np.matmul(beta_tensor[day, stock_idx, :], np.matmul(
                             Q[day, :, :], df_returns.iloc[day + lookback_for_factors])))
                    K[day * stock_idx] = k
                    invest_amount[day + 1] = prod[day * stock_idx]
                continue

            if (day > 0) and (df_score[stock][day] > s_bc) and (state[stock_idx] == 's'):
                day_counter_short[stock_idx] += 1
                if (beta_tensor[day - day_counter_short[stock_idx], stock_idx, :].any(axis=0) == 0) or (Q[day - day_counter_short[stock_idx], :, stock_idx].any(axis=0) == 0):
                    daily_PnL[day, stock_idx] = 0
                else:
                    prod[day * stock_idx] = np.dot(beta_tensor[day - day_counter_short[stock_idx],
                                                   stock_idx, :], Q[day - day_counter_short[stock_idx], :, stock_idx])
                    k = PnL[day - day_counter_short[stock_idx]] * \
                        fraction / (1 + prod[day * stock_idx])

                    factor_contribution.append(k * np.matmul(beta_tensor[day - day_counter_short[stock_idx], stock_idx, :], np.matmul(
                        Q[day - day_counter_short[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                    stock_contribution.append(k * df_returns[stock][day + lookback_for_factors])

                    daily_PnL[day, stock_idx] = k * \
                        (-df_returns[stock][day +
                         lookback_for_factors] + np.matmul(beta_tensor[day - day_counter_short[stock_idx], stock_idx, :], np.matmul(
                             Q[day - day_counter_short[stock_idx], :, :], df_returns.iloc[day + lookback_for_factors])))
                    K[day * stock_idx] = k
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

            # else:
            #     counter_no_trades += 1
            #     continue

        perc_positions[day, 0] = np.count_nonzero(
            state == 'l') / df_score.shape[1]
        perc_positions[day, 1] = np.count_nonzero(
            state == 's') / df_score.shape[1]
        perc_positions[day, 2] = np.count_nonzero(
            state == 'c') / df_score.shape[1]

        fact_cont[day] = sum(factor_contribution)
        stock_cont[day] = sum(stock_contribution)
        fees[day] = (np.abs((invest_amount[day + 1] - invest_amount[day]))).sum() * epsilon

        PnL[day + 1] = PnL[day] + daily_PnL[day, :].sum() - fees[day]

    return PnL, perc_positions, fact_cont, stock_cont, fees


def spy_trading(df_returns, Q):
    df_returns = np.array(df_returns[252:])
    trading_days = df_returns.shape[0]
    n_stocks = df_returns.shape[1]
    A = 1 / Q[0, 0, :].sum()
    # normalized to one, that is the amount of money at time zero
    Q = A * Q[0, 0, :]
    daily_PnL = np.ones(shape=trading_days)
    for i in range(trading_days - 1):
        daily_PnL[i + 1] = daily_PnL[i] + (Q * df_returns[i + 1]).sum()

    return daily_PnL


def grid_search(hyperparameters, score_data, name):
    results = pd.DataFrame(index=range(len(hyperparameters)), columns=[
                           'Hyperparameters', 'SharpeRatio'])
    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass
    for hyper in tqdm(hyperparameters, desc=f'{os.getpid()}'):
        idx = hyperparameters.index(hyper)
        s_bo, s_so, s_bc, s_sc = hyper
        pnl, perc_positions = trading(
            df_returns, df_score, Q, beta_tensor, s_bo=hyper[0], s_so=hyper[1], s_bc=hyper[2], s_sc=hyper[3])
        spy = np.load(
            go_up(1) + '/saved_data/PnL/pnl_FirstPrincipalComp().npy')[:pnl.shape[0]]
        s_ratio = sharpe_ratio(pnl, spy, period=pnl.shape[0])[0]
        print(s_ratio[0], type(s_ratio))
        results['Hyperparameters'][idx] = hyper
        results['SharpeRatio'][idx] = [s_bo, s_so, s_bc, s_sc]
    results.to_csv(
        go_up(1) + f'/saved_data/{name}_{os.getpid()}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('-s', '--spy', action='store_true',
                        help='Generate PnL for the Buy and Hold strategy of the first principal component')
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
    df_returns = pd.read_pickle("/mnt/saved_data/ReturnsDataHugeCleanedWM.pkl")
    # df_returns = pd.read_pickle("/mnt/saved_data/ReturnsData.pkl")
    Q = np.load('/mnt/saved_data/Q_nobias.npy')
    # Q = np.load('/mnt/saved_data/Q.npy')
    length = int(input('Lenght of estimation window for AR(1) parameters: '))
    beta_tensor = np.load(f'/mnt/saved_data/beta_tensor{length}_nobias.npy')
    # beta_tensor = np.load('/mnt/saved_data/beta_tensor60.npy')
    name = input('Name of the s-score data file: ')
    df_score = pd.read_pickle(f'/mnt/saved_data/{name}.pkl')[:4030]
    # df_score = pd.read_pickle(f'/mnt/saved_data/ScoreData_bfgs60_0.pkl')[:4030]

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
        print(hyperparameters[0], len(hyperparameters))

        processes = [mp.Process(target=grid_search, args=(
            hyperparam, df_score, name)) for hyperparam in hyperparameters]
        os.system('rm tmp/*')
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        end = time.time()

        telegram_send.send(messages=[f'GridSearch on {name} terminated'])
        pidnums = [int(x) for x in os.listdir('tmp')]
        pidnums.sort()
        print('AGGIUSTA IL CODICE')
        logging.info('Merging files, then remove the splitted ones...')
        file_merge(pidnums, [name], f'GridSearch_{name}')
        os.system('rm tmp/*')
        exit()

    if args.spy:
        logging.info('Starting spy trading... ')
        time.sleep(1)
        returns = spy_trading(df_returns, Q)
        name = input('Name of the file that will be saved (spy): ')
        np.save(go_up(1) + f'/saved_data/{name}', returns)

    else:
        for i in range(1):
            pnl, perc_positions, fact_cont, stock_cont, fees = trading(
                df_returns, df_score, Q, beta_tensor)
            plt.figure(figsize=(12, 8))
            ax0 = plt.subplot(4, 1, 1)
            ax0.plot(fact_cont, 'k')
            ax1 = plt.subplot(4, 1, 2)
            ax1.plot(stock_cont, 'blue')
            ax2 = plt.subplot(4, 1, 3)
            ax2.plot(fees, 'red')
            ax3 = plt.subplot(4, 1, 4)
            ax3.plot(pnl, 'green')
            plt.show()
            # name = input('Name of the file that will be saved (strategy): ')
            # np.save(go_up(1) + f'/saved_data/PnL/{name}', pnl)
        # name = input('Name of the file that will be saved (positions percentage): ')
        # np.save(go_up(1) + f'/saved_data/{name}', perc_positions)

    end = time.time()
    time_elapsed = (end - start) / 60
    logging.info('Elapsed time: %.2f minutes' % time_elapsed)
