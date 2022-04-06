import argparse
import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import requests
from bs4 import BeautifulSoup
from makedir import go_up
from tqdm import tqdm


def get_ticker():
    """
    Get tickers of companies in S&P500 over all time from Wikipedia page
    url = https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks
    Returns
    ------
    ticker : list
        list of tickers
    """
    website_url = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks').text
    soup = BeautifulSoup(website_url, 'lxml')

    idd_list = ['constituents', 'changes']
    df_list = list()
    for idd in idd_list:
        My_table = soup.find(
            'table', {'class': 'wikitable sortable', 'id': idd})
        df = pd.read_html(str(My_table))
        df = pd.DataFrame(df[0])
        df_list.append(df)

    df_list[1].columns = ['_'.join(col).strip()
                          for col in df_list[1].columns.values]
    df_list[1] = df_list[1].dropna()

    constituents = list(df_list[0].Symbol)
    added = list(df_list[1].Added_Ticker)
    removed = list(df_list[1].Removed_Ticker)

    ticker = list(set(constituents + added + removed))

    return ticker


def price_data(tickers, start, end, data_source='yahoo', export_csv=True):
    '''
    Generate a pandas dataframe of historical close daily price.
    Parameters
    ----------
    start : str
        Start time. Its format must be yyyy-mm-dd
    end : str
        End time. Its format must be yyyy-mm-dd
    data_source : str(optional)
        The data source ("iex", "fred", "ff"). Default = 'yahoo'
    export_csv : bool(optional)
        Choose whether to export to csv. Default = True
    Returns
    -------
    data : pandas.core.frame.DataFrame
        Pandas dataframe of historical close daily price
    '''

    prices = pd.DataFrame(columns=tickers)
    volumes = pd.DataFrame(columns=tickers)
    for tick in tickers:
        try:
            prices[tick] = web.DataReader(
                tick, data_source=data_source, start=start, end=end).Close
            # prices[tick] = data['Close']
            # volumes[tick] = data['Volume']
            print(
                f'{list(tickers).index(tick) + 1}/{len(tickers)} Downloading price and volume data of {tick}')
        except Exception:
            print(
                f"{list(tickers).index(tick) + 1}/{len(tickers)} There is no price or volume data for {tick}")

    # prices = prices.dropna(axis=1)

    if export_csv:
        name = input('Name for the PRICE data file that will be saved: ')
        prices.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)
        # name = input('Name for the VOLUME data file that will be saved: ')
        # volumes.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)

    return prices


def dividends_data(df_price, start, end, export_csv=True):

    dividends = pd.DataFrame()
    # Max request per hour is 500. I have to do this once I dropped some columns from data (without dropping there are 600+ companies)
    for tick in df_price.columns[445:]:
        try:
            dividends[tick] = web.DataReader(
                tick, data_source='tiingo', api_key='77db1f9b52ca2a404420fe0e850ddb042651f945', start=start, end=end).divCash
            logging.info(
                f'{df_price.columns[1:].to_list().index(tick) + 1}/{df_price.columns[1:].shape} Downloading dividend data of {tick}')
        except Exception:
            logging.info(
                f'{df_price.columns[1:].to_list().index(tick) + 1}/{df_price.columns[1:].shape} There is no dividend data for {tick}')

    # dividends = pd.DataFrame(dividends)
    dividends = dividends.fillna(value=0.0)
    # dividends = dividends.drop(columns=['symbol', 'date'])

    if export_csv:
        name = input('Name of the file that will be saved: ')
        dividends.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)

    return dividends


def get_returns(dataframe, volume_integration=False, export_csv=True, m=1):
    """
    Get day-by-day returns values for a company. The dataframe has companies as attributes
    and days as rows, the values are the close prices of each days.

    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame
        Input dataframe of prices.
    export_csv : bool
        Export dataframe in csv.
    m : int (optional)
        Period over which returns are calculated. The default is 1.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        Dataframe of m-period returns.
    """
    try:
        if m < 0:
            raise ValueError(
                "Ops! Invalid input, you can't go backward in time: m must be positive.")
    except ValueError as ve:
        print(ve)
        sys.exit()
    # Drop first column if it is the Date one
    if isinstance(dataframe.iloc[0][0], str):
        dataframe = dataframe.drop(columns=dataframe.columns[0])

    df_ret = pd.DataFrame()
    for col in dataframe.columns:
        df_ret[col] = np.diff(dataframe[col]) / dataframe[col][:-m]

    if volume_integration:
        name = input('Name of volume dataframe')
        df_volume = pd.read_csv(go_up(1) + f'/saved_data/{name}.csv')
        df_vol = pd.DataFrame(index=range(
            df_volume.shape[0] - 10), columns=df_volume.columns)
        for col in tqdm(df_volume.columns, desc='Volume integration'):
            # diff_vol = np.diff(df_volume[col])
            for i in range(df_volume.shape[0] - 10):
                df_vol[col][i] = df_volume[col][i: i + 10].mean() / \
                    df_volume[col][i + 9]
        # Replace nan and inf values with 1. In this way these new returns will be equal to the simple returns
        df_vol.replace([np.inf, -np.inf], 1, inplace=True)
        df_vol = df_vol.fillna(value=1)

        # there are some ticks whose volume data were not available. Have to drop them from returns dataframe
        ticks = ['HFC', 'MXIM', 'XLNX', 'KSU', 'RRD']
        df_ret = pd.DataFrame(df_vol * np.array(df_ret.drop(columns=ticks).iloc[9:]), columns=df_ret.columns)

    if export_returns_csv:
        if volume_integration:
            name = input(
                'Name of the VOLUME INTEGRATED RETURNS file that will be saved: ')
            df_ret.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)
        else:
            name = input(
                'Name of the SIMPLE RETURNS file that will be saved: ')
            df_ret.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)

    return df_ret


def etf_assignment(df_returns, etf):
    df = df_returns
    tickers = df_returns.columns.to_list()
    df_etf = [[pd.read_csv(
        go_up(1) + f'/saved_data/etf/{i}.csv').Ticker.to_list()] for i in etf]
    for tick in tickers:
        for df_etf_columns in df_etf:
            if tick in df_etf_columns[0]:
                # print(tick, etf[df_etf.index(df_etf_columns)])
                df = df.rename(
                    columns={tick: tick + '_' + etf[df_etf.index(df_etf_columns)]})

    file = open(go_up(1) + '/saved_data/no_etf_assignment.txt', 'w')
    c = 0
    for col in df.columns.to_list():
        if '_' not in col:
            c += 1
            file.write(col + '\n')
    file.write(f'Total: {c}')
    file.close()

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument('-s', '--start', type=str,
                        default="1995-01-01", help="Start time")
    parser.add_argument(
        '-end', type=str, default="2020-12-31", help="End time")
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level=levels[args.log])
    price = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv')
    # df_ret = get_returns(price, volume_integration=True)


    SP500histcost = pd.read_csv(go_up(1) + '/saved_data/SP500histcost.csv')
    date_price = price[252:].Date.values
    date_cost = SP500histcost.date.values
    c=0
    df = pd.DataFrame(index=range(date_price.shape[0]), columns=['Date', 'Tickers'])
    for i in range(date_price.shape[0]):
        print(date_price[i], date_cost[i])
        time.sleep(2)
        if date_cost[i] == date_price[i]:
            print('equal')
            df['Date'][i] = date_price[i]
            df['Tickers'][i] = SP500histcost['tickers'][i]
        if date_cost[i] != date_price[i]:
            print('different')
            df['Date'][i] = date_price[i]
            df['Tickers'][i] = SP500histcost['tickers'][i - 1]

    print(df.head(50))
