import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time
import argparse

def data_cleaning(df_returns, limit=2):
    '''This function cleans the returns dataframe removing columns with incosistent data and fill the NaN with zeros. There are some values objectively irrealistic (returns of hundreds or thousands percentage points). '''
    l = []
    for col in ret.columns:
        for val in ret[col]:
            if val > limit:
                print(col)
                time.sleep(1)
                l.append(col)
            if val < -limit:
                l.append(col)
    l = list(set(l))
    ret = ret.drop(columns=l)
    ret = ret.fillna(value=0.0)

    return ret

def costituens_over_time(df_costituens, df_returns, lookback_for_factors=252):
    '''This function plots the number of costituents of S&P500 over the time horizon implied by the
    length of df_costituens/df_returns. Tracks both the total number of tickers, the ones of which
    data were available and downloaded and the ones after the cleaning process on the data.'''

    label_days = [str(x).split(' ')[0] for x in df_returns.index]
    x_label_position = np.arange(0, len(label_days) - lookback_for_factors, lookback_for_factors)
    x_label_day = [label_days[lookback_for_factors + i] for i in x_label_position]

    trading_days = df_returns.shape[0] - lookback_for_factors  # 6294
    num = []
    num1 = []
    num2 = []
    for i in tqdm(range(trading_days), desc='Costituents'):
        oneyear = sp500cost.iloc[i:lookback_for_factors + i]
        period = df_returns[i:lookback_for_factors + i]

        oneyearticks = []
        alloneyearticks = np.unique(oneyear['Tickers'])
        for l_ticks in alloneyearticks:
            for tick in l_ticks:
                oneyearticks.append(tick)
        oneyearticks = list(set(oneyearticks))
        # Take track of costituents during time
        num.append(len(oneyearticks))
        period = period[period.columns.intersection(oneyearticks)]
        # Take track of costituents of which data was available
        num1.append(period.shape[1])
        # Number of costituents without NaN during the 252 days slice
        period = period.dropna(axis=1)
        num2.append(period.shape[1])

    plt.style.use('seaborn')
    plt.figure(figsize=(12,8), tight_layout=True)
    plt.plot(num, linewidth=1.2, alpha=0.75, label='Costituents')
    plt.plot(num1, linewidth=1.2, alpha=0.75, label='Costituents with available data')
    plt.plot(num2, linewidth=1.2, alpha=0.75, label='Costituents after data cleaning')
    plt.ylabel('Number of costituens of S&P500')
    plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
    plt.legend()
    plt.show()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-p", "--plots", action='store_true',
                        help=("Plot returns."))
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Merge files outputed by scoring.py.')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    plt.style.use('seaborn')

    if args.plots:
        lookback_for_factors = 252
        sp500cost = pd.read_pickle('/home/danielemdn/Documents/thesis/StatArb_MS/saved_data/SP500histcost_matched.pkl')
        # df_returns = pd.read_pickle('/home/danielemdn/Documents/thesis/StatArb_MS/saved_data/ReturnsDataHuge.pkl')
        df_returns = pd.read_pickle("/mnt/saved_data/ReturnsDataHugeCleaned.pkl")
        label_days = [str(x).split(' ')[0] for x in df_returns.index]
        x_label_position = np.arange(0, len(label_days) - 252, 252)
        x_label_day = [label_days[252 + i] for i in x_label_position]

        trading_days = df_returns.shape[0] - lookback_for_factors  # 6294
        num = []
        num1 = []
        num2 = []
        for i in tqdm(range(trading_days), desc='Costituents'):
            oneyear = sp500cost.iloc[i:lookback_for_factors + i]
            period = df_returns[i:lookback_for_factors + i]

            oneyearticks = []
            alloneyearticks = np.unique(oneyear['Tickers'])
            for l_ticks in alloneyearticks:
                for tick in l_ticks:
                    oneyearticks.append(tick)
            oneyearticks = list(set(oneyearticks))
            # Take track of costituents during time
            num.append(len(oneyearticks))
            period = period[period.columns.intersection(oneyearticks)]
            # Take track of costituents of which data was available
            num1.append(period.shape[1])
            # Number of costituents without NaN during the 252 days slice
            period = period.dropna(axis=1)
            num2.append(period.shape[1])

        plt.figure(figsize=(12,8), tight_layout=True)
        plt.plot(num, linewidth=1.2, alpha=0.75, label='Costituents')
        plt.plot(num1, linewidth=1.2, alpha=0.75, label='Costituents with available data')
        plt.plot(num2, linewidth=1.2, alpha=0.75, label='Costituents after data cleaning')
        plt.ylabel('Number of costituens of S&P500')
        plt.xticks(x_label_position, x_label_day, fontsize=11,  rotation=90)
        plt.legend()
        plt.show()

    if args.clean:
        # nan_val = []
        # for row in tqdm(range(df_returns.shape[0]), desc='Total NaN'):
        #     c = 0
        #     for element in df_returns.iloc[row]:
        #         if np.isnan(element) : c += 1
        #     nan_val.append(c)
        # plt.figure(figsize=(12,8), tight_layout=True)
        # plt.title('Number of NaN in the entire dataset during time')
        # plt.plot(nan_val, color='darkblue', alpha=0.75)
        # plt.show()


        # sp500cost = pd.read_pickle('../saved_data/SP500histcost_matched.pkl')
        # trading_days = pd.read_pickle('../saved_data/ReturnsDataHuge.pkl').shape[0] - 252
        # ret = pd.read_pickle('/mnt/saved_data/ReturnsDataHuge1.pkl')
        ret = pd.read_pickle('/mnt/saved_data/ReturnsDataHuge.pkl')
        price = pd.read_pickle('/mnt/saved_data/PriceDataHuge.pkl')

        l = []
        for col in ret.columns:
            for val in ret[col]:
                if val > 2.0:
                    l.append(col)
                if val < - 2.0:
                    l.append(col)
        l = list(set(l))
        for tick in l:
            plt.figure(figsize=(11,6))
            plt.plot(price.index, price[tick], linewidth=1, alpha=0.8)
            plt.title(tick)
            plt.show()
        with open('tickers2', 'w', encoding='utf-8') as file:
            file.write(str(l))

        ret = ret.drop(columns=l)
        ret = ret.fillna(value=0.0)
        print(ret.shape)
        # ret.to_pickle('/mnt/saved_data/ReturnsDataHugeCleanedWM1.pkl')
