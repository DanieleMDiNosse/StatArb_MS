import numpy as np
import logging
import argparse
import pandas as pd
from factors import *
from regression_parameters import *
import time
import multiprocessing as mp
import os

def generate_score(df_returns, n_factor, variable_n_factor=False, lookback_for_factors=252, lookback_for_residual=60, export=True):
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
    df_score = df_returns[:trading_days].copy(deep=True) # Il primo score corrisponderà al 252° giorno (indice 251)
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    risk_fact = np.zeros(shape=(trading_days, n_factor))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))

    for i in range(trading_days):
        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        logging.info(f'Trading day: {lookback_for_factors+i}')
        period = df_returns[i:lookback_for_factors + i]
        if variable_n_factor:
            eigenvalues, eigenvectors = pca(
                period, n_components=n_factor, variable_number=variable_n_factor)
        else:
            eigenvalues, eigenvectors = pca(
                period, n_components=n_factor, variable_number=variable_n_factor)
        factors = risk_factors(period, eigenvectors) # ritorni dei fattori di rischio

        risk_fact[i,:] = factors[-1] # mi serve sostanzialmente solamente l'ultimo vettore dei ritorni dei fattori per il trading
        Q[i,:,:] = money_on_stock(period, eigenvectors) # ogni giorno so quanto investire du ogni compagnia all'interno di ognuno dei fattori

        # Ogni giorno i ritorni dei fattori sono valutati su finestre temporali di lunghezza 252 giorni ma che differiscono di un giorno.
        # Devo tener traccia di tutti i ritorni dei fattori di rischio giorno per giorno poiché mi serviranno per il trading.
        # Nota: finora ho utilizzato i fattori di rischio valutati sull'intero dataframe dei ritorni, guardando il futuro!


        # Ottenuti i fattori di rischio si procede con la stima del processo dei residui per ogni compagnia.
        for stock in df_returns.columns:
            # Porre l'intercetta nulla corrisponde a porre il drift alpha nullo
            predictions, beta0, betas, residuals = regression(factors[-lookback_for_residual:], period[-lookback_for_residual:][stock], fit_intercept=False)
            beta_tensor[i, df_returns.columns.get_loc(stock), :] = betas

            discreteOU = np.zeros(len(residuals[-lookback_for_residual:]))
            for j in range(len(discreteOU)):
                discreteOU[j] = residuals[:j + 1].sum()
            discreteOU_predicted, a, b, discrete_residuals = regression(
                discreteOU[:-1].reshape(-1, 1), discreteOU[1:], fit_intercept=True)

            if b == 0.0:
                print(f'B NULLO PER {stock}')
                break

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
        df_score.to_csv(go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)
        np.save(go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
        np.save(go_up(1) + f'/saved_data/risk_fact_{os.getpid()}', risk_fact)
        np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)


def SPY_beta(df_returns, spy, export=True):

    df_returns = np.array(df_returns)
    lookback_for_factors = 252
    lookback_for_residual = 60
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    vec_beta_spy = np.zeros(shape=(trading_days, n_stocks))
    for i in range(trading_days):
        print(f'Trading day: {lookback_for_factors+i}')
        period = df_returns[i:lookback_for_factors + i]
        spy_period = spy[i:lookback_for_factors + i]

        for stock in range(df_returns.shape[1]):
            projection_on_spy, beta0_spy, beta_spy, residuals_spy = regression(
                spy_period[-lookback_for_residual:], period[-lookback_for_residual:, stock], fit_intercept=True)
            vec_beta_spy[i, stock] = beta_spy[0]
    if export:
        name = input('Name of the file that will be saved: ')
        np.save(go_up(1) + f'/saved_data/{name}', vec_beta_spy)

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
    # L_tilde -> Lenght of the slice of the DataFrame, n -> number of slices, L -> total lenght, k -> overlap (i.e. lookback_for_factors-1)
    # L_tilde = (L + k*(n-1))/n
    # with n=5 L_tilde=1510


    start = time.time()
    df_returns = pd.read_csv(go_up(1) +
                             "/saved_data/ReturnsData.csv")
    df = [df_returns[:1510], df_returns[1258:2769], df_returns[2517:4028], df_returns[3776:5287], df_returns[5035:]]
    # # df = [df_returns[:260], df_returns[260:520], df_returns[520:780]]
    #
    processes = [mp.Process(target=generate_score, args=(i, args.n_components)) for i in df]

    for p in processes:
        p.start()
        time.sleep(0.2)

    for p in processes:
        p.join()
        time.sleep(0.2)

    # df_score, beta_tensor = generate_score(df_returns, n_factor=args.n_components, variable_n_factor=args.variable_number,
    #                                        lookback_for_factors=252, lookback_for_residual=60, export=args.save_outputs)

    # spy = pd.read_csv(go_up(1) + '/saved_data/spy.csv')
    # beta_spy = SPY_beta(df_returns, spy, export=args.save_outputs)

    end = time.time()
    time_elapsed = (end - start)
    logging.info('Elapsed time: %.2f seconds' %time_elapsed)
