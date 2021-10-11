import numpy as np
import logging
import argparse
import pandas as pd
from factors import *
from regression_parameters import *
import time
from tqdm import tqdm


def generate_score(df_returns, n_factor, variable_n_factor, lookback_for_factors=252, lookback_for_residual=60, export=False):
    '''This function uses an amount of days equal to lookback_for_factors to evaluate the PCA components and an amount equal to lookback_for_residual
    to evaluate the parameters of the Ornstein-Uhlenbeck process for the residuals. The output is composed by the dataframe of s_score for each stock
    and the beta factors.

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns for each stock.
    n_factor : int
        Number of principal components to keep in the PCA algorithm.
    variable_n_factor : bool
        to be set to True if the number of principal components is chosen through a fixed amount of explained variance, False if it is set by n_factor.
        If it is True, n_factor can be any value: the PCA will not use it.
    lookback_for_factors : int
        Number of days used for the PCA. The default is 252 (one year of trade).
    lookback_for_residual : inf
        Number of days used for the estimation of the residual process (OU). The default is 60.
    export : bool
        Choose whether or not to export to csv the s_score dataframe and to .npy the vector of betas.

    Returns
    -------
    df_score : pandas.core.frame.DataFrame
        Pandas DataFrame of s_score
    beta_tensor : numpy array
        Vector of betas of shape [number of days, number of stock, number of factors]
    '''

    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    df_score = df_returns[:trading_days].copy(deep=True)
    beta_tensor = np.zeros(
        shape=(trading_days, n_stocks, n_factor))
    dev_t = 1 / df_returns.std()

    for i in range(trading_days):
        logging.info(f'Trading day: {lookback_for_factors+i}')
        period = df_returns[i:lookback_for_factors + i]
        if variable_n_factor:
            eigenvalues, eigenvectors = pca(
                period, n_components=n_factor, variable_number=variable_n_factor)
        else:
            eigenvalues, eigenvectors = pca(
                period, n_components=n_factor, variable_number=variable_n_factor)
        factors = risk_factors(period, eigenvectors)

        for stock in df_returns.columns:
            predictions, beta0, betas, residuals = regression(
                factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])
            beta_tensor[i, df_returns.columns.get_loc(stock), :] = betas

            discreteOU = np.zeros(len(residuals[-lookback_for_residual:]))
            for j in range(len(discreteOU)):
                discreteOU[j] = residuals[:j + 1].sum()
            discreteOU_predicted, a, b, discrete_residuals = regression(
                discreteOU[:-1].reshape(-1, 1), discreteOU[1:])
            k = -np.log(b) * lookback_for_factors
            if k < lookback_for_factors / (0.5 * lookback_for_residual):
                df_score[stock][i] = 0
            else:
                m = a / (1 - b)
                sgm = np.std(discrete_residuals) * np.sqrt(2 * k / (1 - b * b))
                sgm_eq = np.std(discrete_residuals) * np.sqrt(1 / (1 - b * b))
                # naive method. Keep in mind that s-score depends on the risk factors
                s_score = -m / sgm_eq
                df_score[stock][i] = s_score[0]

    if export:
        df_score.to_csv(go_up(1) + '/saved_data/ScoreData.csv', index=False)
        np.save(go_up(1) + '/saved_data/beta_tensor', beta_tensor)

    return df_score, beta_tensor


def SPY_beta(df_returns, spy, export=True):

    df_returns = np.array(df_returns)
    lookback_for_factors = 252
    lookback_for_residual = 60
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    vec_beta_spy = np.zeros(shape=(trading_days, n_stocks))
    for i in range(trading_day):
        print(f'Trading day: {lookback_for_factors+i}')
        period = df_returns[i:lookback_for_factors + i]
        spy_period = spy[i:lookback_for_factors + i]

        for stock in range(df_returns.shape[1]):
            projection_on_spy, beta0_spy, beta_spy, residuals_spy = regression(
                spy_period[-lookback_for_residual:], period[-lookback_for_residual:, stock])
            vec_beta_spy[i, stock] = beta_spy[0]
    if export:
        np.save(go_up(1) + '/saved_data/beta_spy1', vec_beta_spy)

    return vec_beta_spy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-vn", "--variable_number", action='store_true',
                        help=("Use a variable number of PCA components. Each time the explained variance is 0.55. The default is False"))
    parser.add_argument('-n', '--n_components', type=int, default=15,
                        help='Number of PCA components to keep. Valid if variable_number is False. The default is 15')
    parser.add_argument('-s', '--save_outputs', action='store_false',
                        help='Choose whether or not to save df_score as csv and beta_tensor as .npy')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    start = time.time()
    df_returns = pd.read_csv(go_up(1) +
                             "/saved_data/ReturnsData.csv")
    # df_score, beta_tensor = generate_score(df_returns, n_factor=args.n_components, variable_n_factor=args.variable_number,
    #                                        lookback_for_factors=252, lookback_for_residual=60, export=args.save_outputs)

    spy = pd.read_csv(go_up(1) + '/saved_data/spy.csv')
    beta_spy = SPY_beta(df_returns, spy, export=args.save_outputs)

    time_elapsed = (end - start)
    logging.info('Elapsed time: %.2f seconds' %time_elapsed)
