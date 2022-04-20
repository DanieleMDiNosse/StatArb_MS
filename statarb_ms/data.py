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
        The data source ("iex", "fred"). Default = 'yahoo'
    export_csv : bool(optional)
        Choose whether to export to csv. Default = True
    Returns
    -------
    data : pandas.core.frame.DataFrame
        Pandas dataframe of historical close daily price
    '''

    prices = pd.DataFrame()
    for tick in tickers:
        try:
            prices[tick] = web.DataReader(
                tick, data_source=data_source, start=start, end=end, api_key='MxXxW-CH1u4v57mxMKh6').Close
            print(
                f'{list(tickers).index(tick) + 1}/{len(tickers)} Downloading price data of {tick}')
        except Exception:
            print(
                f"{list(tickers).index(tick) + 1}/{len(tickers)} There is no price data for {tick}")

    if export_csv:
        name = input('Name for the PRICE data file that will be saved: ')
        prices.to_pickle(go_up(1) + f'/saved_data/{name}.pkl')

    return prices


def get_returns(dataframe, export=True, m=1):
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
    df_ret.index = dataframe.index[1:]

    if export:
        name = input(
            'Name of the SIMPLE RETURNS file that will be saved: ')
        df_ret.to_pickle(go_up(1) + f'/saved_data/{name}.pkl')

    return df_ret


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
    price = pd.read_pickle(go_up(1) + '/saved_data/PriceDataHuge.pkl')
    # df_ret = get_returns(price, volume_integration=True)


    SP500histcost = pd.read_pickle(go_up(1) + '/saved_data/SP500histcost_matched.pkl')
    unique_cost = np.unique(SP500histcost['Tickers'])
    allticks = []
    for costs in unique_cost:
        for tick in costs:
            allticks.append(tick)
    allticks = list(set(allticks))
    # price = price_data(allticks, start='1995-01-01', end='2020-12-31', data_source='quandl', export_csv=True)
    price = pd.read_pickle('/mnt/saved_data/PriceDataHuge1.pkl')
    returns = get_returns(price, volume_integration=False, export=True, m=1)
