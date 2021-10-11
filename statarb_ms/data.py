import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import requests
from bs4 import BeautifulSoup
import logging
import argparse
import json
import time
from makedir import go_up


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

    data = pd.DataFrame()
    for tick in tickers:
        try:
            data[tick] = web.DataReader(
                tick, data_source=data_source, start=start, end=end).Close
            logging.info(f'Downloading price data of {tick}')
        except Exception:
            logging.info(f"There is no price data for {tick}")

    # ATTENZIONE: in questo modo avrò dei giorni vuoti
    data = data.dropna(axis=1)

    if export_csv:
        data.to_csv(go_up(1) + '/saved_data/PriceData.csv')

    return data


def dividends_data(df_price, start, end, export_csv=True):

    dividends = pd.DataFrame()
    c = 0
    # Max request per hour is 500. I have to do this once I dropped some columns from data (without dropping there are 600+ companies)
    for tick in df_price.columns[1:]:
        c += 1
        try:
            dividends[tick] = web.DataReader(
                tick, data_source='tiingo', api_key='49e3a81d8b71b6db04c17fda948be2490d3cdaf9', start=start, end=end).divCash
            logging.info(f'{c} Downloading dividend data of {tick}')
        except Exception:
            logging.info(f'{c} There is no dividend data for {tick}')

    # dividends = pd.DataFrame(dividends)
    dividends = dividends.fillna(value=0.0)
    # dividends = dividends.drop(columns=['symbol', 'date'])

    if export_csv:
        dividends.to_csv(go_up(1) + '/saved_data/DividendsData.csv', index=False)

    return dividends


def get_returns(dataframe, export_returns_csv=True, m=1):
    """
    Get day-by-day returns values for a company. The dataframe has companies as attributes
    and days as rows, the values are the close prices of each days.

    Parameters
    ----------
    dataframe : pandas.core.frame.DataFrame
        Input dataframe of prices.
    export_csv : bool
        Export dataframe in csv.
    standardize : bool
        Use sklearn StandardScaler to scale the returns.
    m : int (optional)
        Period over which returns are calculated. The default is 1.
    no_missing : bool(optional)
        drop companies with missing values. The default is True.

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

    df = pd.DataFrame()
    for col in dataframe.columns:  # Not pick Data column
        today = dataframe[col]
        tomorrow = today[m:]
        df[col] = (np.array(tomorrow) / np.array(today)[:-m]) - 1

    if export_returns_csv:
        name = input('Name of the file that will be saved: ')
        df.to_csv(go_up(1) + f'/saved_data/{name}.csv', index=False)

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
    tickers = get_ticker()
    # data = price_data(tickers,args.start, args.end)
    df_price = pd.read_csv(
        "/home/danielemdn/Documents/thesis/StatArb_MS/saved_data/PriceData.csv")
    # returns = get_returns(df_price)
    div = dividends_data(df_price, start=args.start, end=args.end)
