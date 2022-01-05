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


def trading(df_returns, df_score, df_price, df_dividends, beta_tensor, factors, beta_spy, spy_returns, epsilon=0.0005, s_bo=1.25, s_so=1.25, s_bc=0.75, s_sc=0.50, spy_market_neutrality=True):
    '''This function run a back test of a statistical arbitrage strategy (Avellaneda and Lee, 2010. DOI: 10.1080/14697680903124632).

    Parameters
    ----------
    df_returns :
    '''
    # total_returns = np.ones(df_score.shape[0])
    PnL = np.zeros(df_score.shape[0])
    PnL[0] = 1.0
    lookback_for_factors = 252
    daily_PnL = np.zeros(shape=df_score.shape)
    div = np.zeros(shape=df_score.shape)
    state = np.empty(shape=df_score.shape[1], dtype=str)
    day_counter_long = np.zeros(shape=df_score.shape[1], dtype=int)
    day_counter_short = np.zeros(shape=df_score.shape[1], dtype=int)

    # for day in range(df_score.shape[0] - 1):
    for day in range(df_score.shape[0] - 1):
        logging.info(f'========= Day : {day} =========')
        returns = np.array([])
        for stock in df_score.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            k = PnL[day] * 0.02 / (1 + Q[day,:,stock_idx].sum()) # leverage of portfolio
            logging.info(
                f'-------- Day : {day} Stock : {stock} {stock_idx+1}/{df_score.shape[1]} State : {state[stock_idx]} s_score : {df_score[stock][day]} --------')

            if df_score[stock][day] == 0:
                print(f'Day skipped. Mean reverting speed for {stock} too slow')
                continue

            if df_score[stock][day] < -s_bo and (state[df_returns.columns.get_loc(stock)] != 's'):
                state[stock_idx] = 'l'
                print(f'I have opened a long trade on {stock}')
                daily_PnL[day, stock_idx] = k * (df_returns[stock][day + lookback_for_factors] - (beta_tensor[day, stock_idx, :] * factors[day]).sum())
                continue

            if (day > 0) and (df_score[stock][day] < -s_sc) and (df_score[stock][day] > -s_bo) and (state[df_returns.columns.get_loc(stock)] == 'l'):
                day_counter_long[stock_idx] += 1
                print(f'Still long on {stock}. Long position is opened since {day_counter_long[stock_idx] + 1} days')
                daily_PnL[day, stock_idx] = k * (df_returns[stock][day + lookback_for_factors] - (beta_tensor[day - day_counter_long[stock_idx], stock_idx, :] * factors[day]).sum())
                continue

            if df_score[stock][day] > s_so and (state[df_returns.columns.get_loc(stock)] != 'l'):
                state[df_score.columns.get_loc(stock)] = 's'
                print(f'I have opened a short trade on {stock}')
                daily_PnL[day, stock_idx] = k * (-df_returns[stock][day + lookback_for_factors] + (beta_tensor[day, stock_idx, :] * factors[day]).sum())
                continue

            if (day > 0) and (df_score[stock][day] < s_so) and (df_score[stock][day] > s_bc) and (state[df_returns.columns.get_loc(stock)] == 's'):
                day_counter_short[stock_idx] += 1
                print(f'Still short on {stock}. Long position is opened since {day_counter_short[stock_idx] + 1} days')
                daily_PnL[day, stock_idx] = k * (-df_returns[stock][day + lookback_for_factors] + (beta_tensor[day - day_counter_short[stock_idx], stock_idx, :] * factors[day]).sum())
                continue

            if (day > 0) and (df_score[stock][day] > -s_sc) and (state[df_score.columns.get_loc(stock)] == 'l'):
                print(f'I have closed the trade {state[df_returns.columns.get_loc(stock)]} on {stock}')
                day_counter_long[df_returns.columns.get_loc(stock)] = 0
                state[df_returns.columns.get_loc(stock)] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue

            if (day > 0) and (df_score[stock][day] < s_bc) and (state[df_score.columns.get_loc(stock)] == 's'):
                print(f'I have closed the trade {state[df_returns.columns.get_loc(stock)]} on {stock}')
                day_counter_short[df_returns.columns.get_loc(stock)] = 0
                state[df_returns.columns.get_loc(stock)] = 'c'
                daily_PnL[day, stock_idx] = 0.0
                continue

            else:
                continue

        PnL[day + 1] = PnL[day] + daily_PnL[day,:].sum() - (np.abs(Q[day + 1].sum(axis=1) - Q[day].sum(axis=1))).sum() * epsilon

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
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    start = time.time()
    df_returns = pd.read_csv(go_up(1) + "/saved_data/ReturnsData.csv")
    df_score = pd.read_csv(go_up(1) + '/saved_data/ScoreData.csv') # ottenuti unendo quelli dei processi
    df_price = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv')
    df_dividends = pd.read_csv(go_up(1) + '/saved_data/DividendsData.csv')
    beta_tensor = np.load(go_up(1) + '/saved_data/Russel_beta_tensor.npy')
    factors = np.load(go_up(1) + '/saved_data/Russel_risk_factors.npy') # ottenuti unendo quelli dei processi
    beta_spy = np.load(go_up(1) + '/saved_data/beta_spy.npy')
    spy_returns = pd.read_csv(go_up(1) + '/saved_data/ReturnsSPY.csv')
    Q = np.load(go_up(1) + '/saved_data/RusselQ.npy')

    returns = trading(df_returns, df_score, df_price,
                      df_dividends, beta_tensor, factors, beta_spy, spy_returns)
    name = input('Name of the file that will be saved: ')
    np.save(go_up(1) + f'/saved_data/{name}', returns)

    end = time.time()
    time_elapsed = (end - start)
    logging.info('Elapsed time: %.2f minutes' % time_elapsed/60)
