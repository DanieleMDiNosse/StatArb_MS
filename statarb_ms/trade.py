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


def trading(df_returns, df_score, df_price, df_dividends, beta_tensor, factors, beta_spy, spy_returns, s_bo=1.25, s_so=1.25, s_bc=0.75, s_sc=0.50, spy_market_neutrality=True):
    '''This function run a back test of a statistical arbitrage strategy (Avellaneda and Lee, 2010. DOI: 10.1080/14697680903124632).

    Parameters
    ----------
    df_returns :
    '''
    total_returns = np.ones(df_score.shape[0])
    total_returns[0] = 1000
    lookback_for_factors = 252
    long_stocks = np.zeros(shape=df_score.shape)
    long_factors = np.zeros(shape=df_score.shape)
    short_stocks = np.zeros(shape=df_score.shape)
    short_factors = np.zeros(shape=df_score.shape)
    div = np.zeros(shape=df_score.shape)
    state = np.empty(shape=df_score.shape[1], dtype=str)

    for day in range(df_score.shape[0] - 1):
        logging.info(f'========= Day : {day} =========')
        returns = np.array([])
        n = 0
        beta = 0
        for stock in df_score.columns:
            logging.info(
                f'-------- Day : {day} Stock : {stock} {df_score.columns.get_loc(stock)+1}/{df_score.shape[1]} State : {state[df_score.columns.get_loc(stock)]} s_score : {df_score[stock][day]} --------')

            if df_score[stock][day] == 0:
                print(f'mean reverting speed for {stock} too slow')
                pass

            if df_score[stock][day] < -s_bo and (state[df_returns.columns.get_loc(stock)] != 's'):
                state[df_score.columns.get_loc(stock)] = 'l'
                n += 1
                beta += beta_spy[day + 1, df_score.columns.get_loc(stock)]
                print(f'I have opened a long trade on {stock}')

                long_stocks[day][df_returns.columns.get_loc(
                    stock)] = df_returns[stock][day + lookback_for_factors + 1]
                div[day][df_returns.columns.get_loc(
                    stock)] = df_dividends[stock][day + lookback_for_factors + 1] / df_price[stock][day + lookback_for_factors + 1]
                if spy_market_neutrality:
                    short_factors[day][df_returns.columns.get_loc(
                        stock)] = -beta_spy[day + 1, df_score.columns.get_loc(stock)] * spy_returns.iloc[day + lookback_for_factors + 1]
                else:
                    short_factors[day][df_returns.columns.get_loc(
                        stock)] = (-factors[day + lookback_for_factors + 1] * beta_tensor[day + 1, df_score.columns.get_loc(stock), :]).sum()

            if (day > 0) and (df_score[stock][day] < -s_sc) and (df_score[stock][day] > -s_bo) and (state[df_returns.columns.get_loc(stock)] == 'l'):
                print(f'Still long on {stock}')

                long_stocks[day][df_returns.columns.get_loc(
                    stock)] = df_returns[stock][day + lookback_for_factors + 1]
                div[day][df_returns.columns.get_loc(
                    stock)] = df_dividends[stock][day + lookback_for_factors + 1] / df_price[stock][day + lookback_for_factors + 1]
                if spy_market_neutrality:
                    short_factors[day][df_returns.columns.get_loc(
                        stock)] = -beta_spy[day + 1, df_score.columns.get_loc(stock)] * spy_returns.iloc[day + lookback_for_factors + 1]
                else:
                    short_factors[day][df_returns.columns.get_loc(
                        stock)] = (-factors[day + lookback_for_factors + 1] * beta_tensor[day + 1, df_score.columns.get_loc(stock), :]).sum()

            if df_score[stock][day] > s_so and (state[df_returns.columns.get_loc(stock)] != 'l'):
                state[df_score.columns.get_loc(stock)] = 's'
                n += 1
                beta += beta_spy[day + 1, df_score.columns.get_loc(stock)]
                print(f'I have opened a short trade on {stock}')

                short_stocks[day][df_returns.columns.get_loc(
                    stock)] = -df_returns[stock][day + lookback_for_factors + 1]
                div[day][df_returns.columns.get_loc(
                    stock)] = -df_dividends[stock][day + lookback_for_factors + 1] / df_price[stock][day + lookback_for_factors + 1]
                if spy_market_neutrality:
                    long_factors[day][df_returns.columns.get_loc(
                        stock)] = beta_spy[day + 1, df_score.columns.get_loc(stock)] * spy_returns.iloc[day + lookback_for_factors + 1]
                else:
                    long_factors[day][df_returns.columns.get_loc(stock)] = (
                        factors[day + lookback_for_factors + 1] * beta_tensor[day + 1, df_score.columns.get_loc(stock), :]).sum()

            if (day > 0) and (df_score[stock][day] < s_so) and (df_score[stock][day] > s_bc) and (state[df_returns.columns.get_loc(stock)] == 's'):
                print(f'Still short on {stock}')

                short_stocks[day][df_returns.columns.get_loc(
                    stock)] = -df_returns[stock][day + lookback_for_factors + 1]
                div[day][df_returns.columns.get_loc(
                    stock)] = -df_dividends[stock][day + lookback_for_factors + 1] / df_price[stock][day + lookback_for_factors + 1]
                if spy_market_neutrality:
                    long_factors[day][df_returns.columns.get_loc(
                        stock)] = beta_spy[day + 1, df_score.columns.get_loc(stock)] * spy_returns.iloc[day + lookback_for_factors + 1]
                else:
                    long_factors[day][df_returns.columns.get_loc(stock)] = (
                        factors[day + lookback_for_factors + 1] * beta_tensor[day + 1, df_score.columns.get_loc(stock), :]).sum()

            if (day > 0) and (df_score[stock][day] < s_bc) and (df_score[stock][day] > -s_sc):
                if state[df_score.columns.get_loc(stock)] == 'l':
                    print(
                        f'I have closed the trade {state[df_returns.columns.get_loc(stock)]} on {stock}')
                    state[df_returns.columns.get_loc(stock)] = 'c'

                    profit_long_stocks = long_stocks[:, df_returns.columns.get_loc(stock)].sum()
                    long_stocks[:, df_returns.columns.get_loc(
                        stock)] = np.zeros(shape=df_score.shape[0])

                    profit_short_factors = short_factors[:, df_returns.columns.get_loc(
                        stock)].sum()
                    short_factors[:, df_returns.columns.get_loc(
                        stock)] = np.zeros(shape=df_score.shape[0])

                    div_profit = div[:,
                                     df_returns.columns.get_loc(stock)].sum()
                    div[:, df_returns.columns.get_loc(stock)] = np.zeros(
                        shape=df_score.shape[0])

                    returns = np.append(
                        returns, profit_long_stocks + profit_short_factors + div_profit)

                if state[df_returns.columns.get_loc(stock)] == 's':
                    print(
                        f'I have closed the trade {state[df_returns.columns.get_loc(stock)]} on {stock}')
                    state[df_returns.columns.get_loc(stock)] = 'c'

                    profit_short_stocks = short_stocks[:, df_returns.columns.get_loc(
                            stock)].sum()
                    short_stocks[:, df_returns.columns.get_loc(
                        stock)] = np.zeros(shape=df_score.shape[0])

                    profit_long_factors = short_factors[:, df_returns.columns.get_loc(
                        stock)].sum()
                    long_factors[:, df_returns.columns.get_loc(
                        stock)] = np.zeros(shape=df_score.shape[0])

                    div_profit = div[:,
                                     df_returns.columns.get_loc(stock)].sum()
                    div[:, df_returns.columns.get_loc(stock)] = np.zeros(
                        shape=df_score.shape[0])

                    returns = np.append(
                        returns, profit_short_stocks + profit_long_factors + div_profit)

            else:
                pass

        total_returns[day + 1] = total_returns[day] + returns.sum()

    return total_returns


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
    df_returns = pd.read_csv(go_up(1) +
                             "/saved_data/RusselReturnsData.csv")
    df_score = pd.read_csv(go_up(1) +
                           "/saved_data/ScoreData.csv")
    df_price = pd.read_csv(go_up(1) + '/saved_data/RusselPriceData.csv')
    df_dividends = pd.read_csv(go_up(1) + '/saved_data/RusselDividendsData.csv')
    beta_tensor = np.load(go_up(1) + '/saved_data/beta_tensor.npy')
    factors = np.load(go_up(1) + '/saved_data/Russelfactors.npy')
    beta_spy = np.load(go_up(1) + '/saved_data/russel_beta_spy.npy')
    spy_returns = pd.read_csv(go_up(1) + '/saved_data/ReturnsSPY.csv')

    returns = trading(df_returns, df_score, df_price,
                      df_dividends, beta_tensor, factors, beta_spy, spy_returns, spy_market_neutrality=False)
    name = input('Name of the file that will be saved: ')
    np.save(go_up(1) + f'/saved_data/{name}', returns)

    end = time.time()
    time_elapsed = (end - start)
    logging.info('Elapsed time: %.2f seconds' % time_elapsed)
