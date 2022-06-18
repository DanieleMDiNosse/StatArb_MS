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
from tqdm import tqdm


def get_ticker():
    """
    Get tickers of costituents of S&P500 over all time available from the Wikipedia page
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks'

    Returns
    -------
    ticker : list
        List of strings (tickers).
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


def price_data(tickers, start, end, data_source='yahoo', export=True):
    '''
    Generate a pandas dataframe of historical close daily prices based on the pandas_datareader package.

    Parameters
    ----------
    tickers: list
        List of strings (tickers).
    start : str
        Start time. Format must be yyyy-mm-dd
    end : str
        End time. Format must be yyyy-mm-dd
    data_source : str (optional)
        Data source. For all the possible options see pandas_datareader docs at 'https://pydata.github.io/pandas-datareader/remote_data.html'. Default is 'yahoo'.
    export : bool (optional)
        Choose whether to export to pandas pickle format. Default is True.

    Returns
    -------
    prices : pandas.core.frame.DataFrame
        Pandas dataframe of historical close daily price.
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
    Generate m-period returns for all the companies in dataframe.columns. Dataframe has tickers as atributes and daily close prices as entries.

    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame
        Input dataframe of daily close prices.
    export : bool (optional)
        Choose whether to export to pandas pickle format. Default is True.
    m : int (optional)
        Period over which returns are calculated. The default is 1.

    Returns
    -------
    df_ret : pandas.core.frame.DataFrame
        Dataframe of m-period returns.
    """

    try:
        if m < 0:
            raise ValueError(
                "Ops! Invalid input: m must be positive.")
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
            'Name of the m-period returns file that will be saved: ')
        df_ret.to_pickle(go_up(1) + f'/saved_data/{name}.pkl')

    return df_ret


def volume_integration(df_volume, df_returns, lookback=10, export=True):
    '''This function transforms the dataframe of returns df_returns into a new one in which the (i,j) element is the return of stock j at day i wheighted by the ratio between the average daily volume changes over lookback days and the last daily volume change.

    Paramters
    ---------
    df_volume : pandas.core.frame.DataFrame
        Dataframe of daily volume traded.
    df_returns : pandas.core.frame.DataFrame
        Dataframe of returns.
    lookback : int (optional)
        Period over which the average daily volume traded is computed. The default is 10.
    export : bool (optional)
        Choose if the outcome dataframe is saved (True) or not (False). The default is True.

    Returns
    -------
    vol_int_ret : pandas.core.frame.DataFrame
        Volume weighted dataframe.
    '''

    # clean volume data
    df_volume = df_volume.dropna(axis=1)

    # create a dataframe of lookback averages of daily differences volume
    vol_per_mean = pd.DataFrame(index=range(
        df_volume.shape[0] - lookback), columns=df_volume.columns)
    for col in tqdm(df_volume.columns, desc='Average volume'):
        for day in range(vol_per_mean.shape[0]):
            vol_per = df_volume[col][day: day + lookback]
            vol_per_mean[col][day] = vol_per.mean()

    # Take the inverse and multiply it by the average daily volume over lookback days
    df_volume = df_volume.apply(lambda x: 1 / x)
    # The first needed value of the dataframe of daily volume differences is the lookback-th
    df_volume = df_volume[lookback:]
    df_volume.index = range(vol_per_mean.shape[0])
    vol_weights = df_volume * vol_per_mean
    # if there is inf or -inf, replace it with 1. These entries will not contribute to the weighting procedure on the returns.
    vol_weights = vol_weights.replace([np.inf, -np.inf], 1)
    vol_weights = vol_weights.fillna(1)

    df_ret = df_returns[lookback - 1:]
    df_ret.index = range(df_ret.shape[0])
    vol_int_ret = df_ret * vol_weights
    # vol_int_ret = vol_int_ret[1:-1]

    if export:
        vol_int_ret.to_pickle('/mnt/saved_data/ReturnsVolData.pkl')
        vol_weights.to_pickle('/mnt/saved_data/VolumeWeights.pkl')
        vol_per_mean.to_pickle('/mnt/saved_data/VolPerMean.pkl')

    return vol_int_ret


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
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    df_volume = pd.read_pickle('/mnt/saved_data/VolumeData.pkl')
    df_returns = pd.read_pickle(
        '/mnt/saved_data/ReturnsData.pkl')[df_volume.columns.to_list()]

    vol_int = volume_integration(df_volume, df_returns, export=True)
