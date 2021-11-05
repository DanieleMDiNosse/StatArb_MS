import numpy as np
import logging
import argparse
import pandas as pd
from factors import *
from regression_parameters import *
import time
from tqdm import tqdm
from scoring import *
from makedir import *


def trading(df_returns, df_score, Q, beta_tensor, epsilon=0.0005, s_bo=1.25, s_so=1.25, s_bc=0.75, s_sc=0.50):
    '''This function run a back test of a statistical arbitrage strategy (Avellaneda and Lee, 2010. DOI: 10.1080/14697680903124632).

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
    PnL[0] = 1.0
    lookback_for_factors = 252
    daily_PnL = np.zeros(shape=df_score.shape)
    div = np.zeros(shape=df_score.shape)
    state = np.array(['c' for i in range(df_score.shape[1])])
    day_counter_long = np.zeros(shape=df_score.shape[1], dtype=int)
    day_counter_short = np.zeros(shape=df_score.shape[1], dtype=int)

    days_opened = []

    for day in range(df_score.shape[0] - 1):
        logging.info(f'========= Day : {day} =========')
        returns = np.array([])
        for stock in df_score.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            k = PnL[day] * 0.01 / (1 + (beta_tensor[day, stock_idx, :] * Q[day,:,stock_idx]).sum()) # leverage of portfolio
            logging.info(
                f'-------- Day : {day} Stock : {stock} {stock_idx+1}/{df_score.shape[1]} State : {state[stock_idx]} s_score : {df_score[stock][day]} --------')

            if df_score[stock][day] == 0:
                print(f'Day skipped. Mean reverting speed for {stock} too slow')
                continue

            if df_score[stock][day] < -s_bo and (state[stock_idx] == 'c'):
                state[stock_idx] = 'l'
                print(f'I have opened a long trade on {stock}')
                daily_PnL[day, stock_idx] = k * df_returns[stock][day + lookback_for_factors] * (1 - (beta_tensor[day, stock_idx, :] * Q[day,:,stock_idx]).sum())
                continue

            if (day > 0) and (df_score[stock][day] < -s_sc) and (state[stock_idx] == 'l'):
                day_counter_long[stock_idx] += 1
                print(f'Still long on {stock}. Long position is opened since {day_counter_long[stock_idx] + 1} days')
                daily_PnL[day, stock_idx] = k * df_returns[stock][day + lookback_for_factors] * (1 - (beta_tensor[day - day_counter_long[stock_idx], stock_idx, :] * Q[day - day_counter_long,:,stock_idx]).sum())
                continue

            if df_score[stock][day] > s_so and (state[stock_idx] == 'c'):
                state[stock_idx] = 's'
                print(f'I have opened a short trade on {stock}')
                daily_PnL[day, stock_idx] = k * df_returns[stock][day + lookback_for_factors] * (-1 + (beta_tensor[day, stock_idx, :] * Q[day,:,stock_idx]).sum())
                continue

            if (day > 0) and (df_score[stock][day] > s_bc) and (state[stock_idx] == 's'):
                day_counter_short[stock_idx] += 1
                print(f'Still short on {stock}. Short position is opened since {day_counter_short[stock_idx] + 1} days')
                daily_PnL[day, stock_idx] = k * df_returns[stock][day + lookback_for_factors] * (-1 + (beta_tensor[day - day_counter_short[stock_idx], stock_idx, :] * Q[day - day_counter_short,:,stock_idx]).sum())
                continue

            if (day > 0) and (df_score[stock][day] > -s_sc) and (state[stock_idx] == 'l'):
                print(f'I have closed the trade {state[stock_idx]} on {stock}')
                days_opened.append(day_counter_long[stock_idx])
                day_counter_long[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue

            if (day > 0) and (df_score[stock][day] < s_bc) and (state[stock_idx] == 's'):
                print(f'I have closed the trade {state[stock_idx]} on {stock}')
                days_opened.append(day_counter_short[stock_idx])
                day_counter_short[stock_idx] = 0
                state[stock_idx] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue

            else:
                continue

        PnL[day + 1] = PnL[day] + daily_PnL[day,:].sum() - (np.abs(Q[day + 1].sum(axis=1) - Q[day].sum(axis=1))).sum() * epsilon

    days_opened = np.array(days_opened)
    with open(go_up(1) + '/saved_data/summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'On average a long/short position in opened for {days_opened.mean()} +/- {days_opened.std()/days_opened.shape} days')
    return PnL


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-vn", "--variable_number", action='store_true',
                        help=("Use a variable number of PCA components. Each time the explained variance is 0.55. The default is False"))
    parser.add_argument('-n', '--n_components', type=int, default=15,
                        help='Number of PCA components to keep. Valid if variable_number is False. The default is 15')
    parser.add_argument('-r', '--russel', action='store_true')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    start = time.time()
    if args.russel:
        df_returns = pd.read_csv(go_up(1) + "/saved_data/RusselReturnsData.csv")
        df_score = pd.read_csv(go_up(1) + '/saved_data/RusselScoreData.csv') # ottenuti unendo quelli dei processi
        # df_price = pd.read_csv(go_up(1) + '/saved_data/RusselPriceData.csv')
        # df_dividends = pd.read_csv(go_up(1) + '/saved_data/DividendsData.csv')
        beta_tensor = np.load(go_up(1) + '/saved_data/Russel_beta_tensor.npy')
        # factors = np.load(go_up(1) + '/saved_data/Russel_risk_factors.npy') # ottenuti unendo quelli dei processi
        # beta_spy = np.load(go_up(1) + '/saved_data/beta_spy.npy')
        # spy_returns = pd.read_csv(go_up(1) + '/saved_data/ReturnsSPY.csv')
        Q = np.load(go_up(1) + '/saved_data/RusselQ.npy')
    else:
        df_returns = pd.read_csv(go_up(1) + "/saved_data/ReturnsData.csv")
        df_score = pd.read_csv(go_up(1) + '/saved_data/ScoreData.csv') # ottenuti unendo quelli dei processi
        beta_tensor = np.load(go_up(1) + '/saved_data/beta_tensor.npy')
        Q = np.load(go_up(1) + '/saved_data/Q.npy')


    returns = trading(df_returns, df_score, Q, beta_tensor)
    name = input('Name of the file that will be saved: ')
    np.save(go_up(1) + f'/saved_data/{name}', returns)

    end = time.time()
    time_elapsed = (end - start)/60
    logging.info('Elapsed time: %.2f minutes' % time_elapsed)
