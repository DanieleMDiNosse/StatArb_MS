import numpy as np
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from makedir import go_up
from post_processing import file_merge, remove_file
from reduced_loglikelihood import reduced_loglikelihood
from gas import estimation
from factors import pca, risk_factors, money_on_stock
from regression_parameters import regression, auto_regression
from sklearn.metrics import r2_score
import time
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from numba import njit


def generate_data(df_returns, n_factor, method, lookback_for_factors=252, lookback_for_residual=60, export=True):
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
    discreteOU : numpy ndarray
        Discrete version of the Ornstein Uhlenbeck process. This array will be used to estimate s-scores.
    beta_tensor : numpy ndarray
        Weights of each of hte pca components for each day and each stock. Dimensions are (trading days x n_stocks x n_factors).
    Q : numpy ndarray
        Money to invest on each factors for each days. Dimensions are dim (trading days x n_factors x n_components x n_stocks)
    '''

    trading_days = df_returns.shape[0] - lookback_for_factors # 6294
    trading_days = 500
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    score = np.zeros(shape=(trading_days, n_stocks)) # Il primo score corrisponderà al 252° giorno (indice 251)
    b_values = np.zeros(shape=(trading_days, n_stocks, 3))
    R_squared = np.zeros(shape=(trading_days, n_stocks))
    dis_res_reg = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days)):
        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        period = df_returns[i:lookback_for_factors + i] # [0,252[, [1,253[ ecc -> ogni period comprende un anno di trading (252 giorni)
        eigenvalues, eigenvectors = pca(period, n_components=n_factor)
        factors = risk_factors(period, eigenvectors)# ritorni dei fattori di rischio per ogni periodo
        Q[i,:,:] = money_on_stock(period, eigenvectors)# trading_days x n_factors x n_stocks. Ogni giorno so quanto investire su ogni compagnia all'interno di ognuno dei fattori

        # Ottenuti i fattori di rischio si procede con la stima del processo dei residui per ogni compagnia.
        for stock in df_returns.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            beta0, betas, conf_inter, residuals, pred, _ = regression(factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])
            beta_tensor[i, stock_idx, :] = betas

            discreteOU = np.cumsum(residuals)
            dis_res[i, stock_idx, :] = discreteOU

            if method == 'constant_speed':
                discreteOU = np.append(discreteOU, discreteOU[-1])
                parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(discreteOU)
                a, b = parameters[0], parameters[1]
                # Registro la dinamica di b nel tempo, memorizzando media e intervallo di confidenza. L'idea successiva è quella di modellizzare b
                # tramite un GAS model.
                b_values[i, stock_idx, 0], b_values[i, stock_idx, 1:] = b, discrete_conf_int[1,:]
                R_squared[i, stock_idx] = r2_score(discreteOU[:-1], np.array(discrete_pred))
                # Registro tutti i residui per i test sulla normalità ed "indipendenza" da fare successivamente
                dis_res_reg[i, stock_idx, :] = discrete_resid

            if method == 'gas_modelization':
                b, a, xi = estimation(reduced_loglikelihood, discreteOU, method='Nelder-Mead', update='gaussian', verbose=False)
                b = b[-1] # Mi serve solo l'ultimo valore per lo score
                if b < 0: # se b è negativo, sostituiscilo con quello stimato supponendo sia costante nella finestra temporale
                    discreteOU = np.append(discreteOU, discreteOU[-1])
                    parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(discreteOU)
                    a, b = parameters[0], parameters[1]
                # devi aggiungere anche le bande di confidenza

            if b == 0.0:
                print(f'B NULLO PER {stock}')
                break

            k = -np.log(b) * lookback_for_factors
            if k < lookback_for_factors / (0.5 * lookback_for_residual):
                score[i, stock_idx] = 0
            else:
                m = a / (1 - b)
                # sgm = np.std(discrete_resid) * np.sqrt(2 * k / (1 - b * b))
                if method == 'constant_speed': sgm_eq = np.std(discrete_resid) * np.sqrt(1 / (1 - b * b))
                if method == 'gas_modelization': sgm_eq = np.std(xi) * np.sqrt(1 / (1 - b * b))
                # naive method. Keep in mind that s-score depends on the risk factors
                s_score = -m / sgm_eq
                score[i, stock_idx] = s_score

    df_score = pd.DataFrame(score, columns=df_returns.columns)
    if export:
        if method == 'gas_modelization':
            df_score.to_csv(go_up(1) + f'/saved_data/df_score_gas_{os.getpid()}.csv', index=False)
            np.save(go_up(1) + f'/saved_data/dis_res_reg_gas_{os.getpid()}', dis_res_reg)
            np.save(go_up(1) + f'/saved_data/b_values_gas_{os.getpid()}', b_values)
        if method == 'constant_speed':
            df_score.to_csv(go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)
            np.save(go_up(1) + f'/saved_data/dis_res_reg_{os.getpid()}', dis_res_reg)
            np.save(go_up(1) + f'/saved_data/b_values_{os.getpid()}', b_values)
            np.save(go_up(1) + f'/saved_data/R_squared_{os.getpid()}', R_squared)
            np.save(go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
            np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)
            np.save(go_up(1) + f'/saved_data/dis_res_{os.getpid()}', dis_res)

# def generate_score(df_returns, dis_res, method, lookback_for_factors=252, lookback_for_residual=100, export=True):
#     '''Compute the s-scores from the outputs of projection function.
#
#     Parameters
#     ----------
#     df_returns : pandas.core.frame.DataFrame
#         Dataframe of 1-day returns for each stock.
#     dis_res : numpy ndarray
#         Discrete version of the Ornstein Uhlenbeck process.
#     method : str
#         Choose whether to use GAS estimation ('gas_modelization') with b=b(t) or constant b modelling with autoregression
#         estimation ('constant_speed').
#     lookback_for_factors : int
#         Number of days used for the PCA. The default is 252 (one year of trade).
#     lookback_for_residual : inf
#         Number of days used for the estimation of the residual process (OU). The default is 60.
#     export : bool
#         Choose whether or not to export to csv the s_score dataframe and to .npy the vector of betas.
#
#     Returns
#     -------
#     df_score : pandas.core.frame.DataFrame
#         Dataframe of s-scores for each stock and each day.
#     b_values:  numpy ndarray
#         Values of b with confidence intervals. Dimension are (trading_days, n_stocks, 3). First value is b, the last two represents
#         the confidence intervals.
#     R_squared: numpy ndarray
#         Coefficient of deterination between dis_res and the estimated series with gas/autoregression.
#     dis_res_reg: numpy ndarray
#         Array of residuals of the estimation procedure.
#     '''
#
#     trading_days = df_returns.shape[0] - lookback_for_factors
#     n_stocks = df_returns.shape[1]
#
#     score = np.zeros(shape=(trading_days, n_stocks)) # Il primo score corrisponderà al 252° giorno (indice 251)
#     b_values = np.zeros(shape=(trading_days, n_stocks, 3))
#     R_squared = np.zeros(shape=(trading_days, n_stocks))
#     dis_res_reg = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
#
#     with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
#         pass
#
#     for i in tqdm(range(trading_days)):
#         for stock in df_returns.columns:
#             stock_idx = df_returns.columns.get_loc(stock)
#             if method == 'constant_speed':
#                 discreteOU = np.append(dis_res[i, stock_idx, :], dis_res[i, stock_idx, :][-1])
#                 parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(discreteOU, mod='cmle', diagnostics=False)
#                 a, b = parameters[0], parameters[1]
#                 # Registro la dinamica di b nel tempo, memorizzando media e intervallo di confidenza. L'idea successiva è quella di modellizzare b
#                 # tramite un GAS model.
#                 b_values[i, stock_idx, 0], b_values[i, stock_idx, 1:] = b, discrete_conf_int[1,:]
#                 R_squared[i, stock_idx] = r2_score(discreteOU[:-1], np.array(discrete_pred))
#
#             if method == 'gas_modelization':
#                 omega, a, alpha, beta = estimation(reduced_loglikelihood, dis_res[i, stock_idx, :], method='Nelder-Mead', verbose=False)
#                 b, discrete_resid = recostruction(dis_res[i, stock_idx, :], omega, a, alpha, beta)
#                 b = b[-1] # Mi serve solo l'ultimo valore per lo score
#                 if b < 0: # se b è negativo, sostituiscilo con quello stimato supponendo sia costante nella finestra temporale
#                     discreteOU = np.append(dis_res[i, stock_idx, :], dis_res[i, stock_idx, :][-1])
#                     parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(discreteOU, mod='cmle', diagnostics=False)
#                     a, b = parameters[0], parameters[1]
#                 # devi aggiungere anche le bande di confidenza
#             # Registro tutti i residui per i test sulla normalità ed "indipendenza" da fare successivamente
#             dis_res_reg[i, stock_idx, :] = discrete_resid
#
#             if b == 0.0:
#                 print(f'B NULLO PER {stock}')
#                 break
#
#             k = -np.log(b) * lookback_for_factors
#             if k < lookback_for_factors / (0.5 * lookback_for_residual):
#                 score[i, stock_idx] = 0
#             else:
#                 m = a / (1 - b)
#                 sgm = np.std(discrete_resid) * np.sqrt(2 * k / (1 - b * b))
#                 sgm_eq = np.std(discrete_resid) * np.sqrt(1 / (1 - b * b))
#                 # naive method. Keep in mind that s-score depends on the risk factors
#                 s_score = - m / sgm_eq
#                 score[i, stock_idx] = s_score
#     df_score = pd.DataFrame(score, columns=df_returns.columns)
#
#     if export:
#         if method == 'gas_modelization':
#             df_score.to_csv(go_up(1) + f'/saved_data/df_score_gas_{os.getpid()}.csv', index=False)
#             np.save(go_up(1) + f'/saved_data/dis_res_reg_gas_{os.getpid()}', dis_res_reg)
#             np.save(go_up(1) + f'/saved_data/b_values_gas_{os.getpid()}', b_values)
#         if method == 'constant_speed':
#             df_score.to_csv(go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)
#             np.save(go_up(1) + f'/saved_data/dis_res_reg_{os.getpid()}', dis_res_reg)
#             np.save(go_up(1) + f'/saved_data/b_values_{os.getpid()}', b_values)
#             np.save(go_up(1) + f'/saved_data/R_squared_{os.getpid()}', R_squared)
#
#     return df_score, b_values, R_squared, dis_res_reg


def SPY_beta(df_returns, spy, lookback_for_factors=252, lookback_for_residual=60, export=True):

    df_returns = np.array(df_returns)
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
                        help='Number of PCA components to keep. The default is 15.')
    parser.add_argument("-g", "--gas", action='store_true',
                        help=("Use gas estimation for the mean reverting speed. The default is False."))
    parser.add_argument('-s', '--save_outputs', action='store_false',
                        help='Choose whether or not to save the outputrs. The default is True')
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
    # df = [df_returns[768:1789], df_returns[1537:2558], df_returns[2306:3327], df_returns[3075:4096]]
    # df = [df_returns[:1020], df_returns[768:1789], df_returns[1537:2558], df_returns[2306:3327], df_returns[3075:4096], df_returns[3844:4865], df_returns[4613:5634], df_returns[5382:]]
    df = [df_returns[:1510], df_returns[1258:2769], df_returns[2517:4028], df_returns[3776:5287], df_returns[5035:]]

    if args.gas == True:
        method = 'gas_modelization'
    else:
        method = 'constant_speed'

    processes = [mp.Process(target=generate_data, args=(i, args.n_components, method, 252, 60, args.save_outputs)) for i in df]
    os.system('rm tmp/*')
    for p in processes:
        p.start()
        time.sleep(0.5)
    for p in processes:
        p.join()

    end = time.time()


    pidnums = [int(x) for x in os.listdir('tmp')]
    pidnums.sort()
    if args.gas:
        file_list = ['beta_tensor', 'Q', 'dis_res', 'df_score_gas', 'dis_res_reg_gas', 'b_values_gas']
    else:
        file_list = ['beta_tensor', 'Q', 'dis_res', 'df_score', 'dis_res_reg', 'b_values', 'R_squared']
    logging.info('Merging files...')
    file_merge(pidnums, file_list)
    remove_file(pidnums, file_list)
    os.system('rm tmp/*')
    # for x in os.listdir('tmp'):
    #     os.remove(f'tmp/{x}')

    # keep_going = str(input('Do you want to continue and compute the s-scores? [y/n]: '))
    #
    # if (keep_going == 'y'):
    #
    #     logging.info('I am going to estimate the parameters of the OU process and generate the s-scores')
    #     time.sleep(1)
    #
    #     name = str(input('Name you used for dis_res: '))
    #     dis_res = np.load(go_up(1) + f'/saved_data/{name}.npy')
    #     # df_score, b_values, R_squared, dis_res_reg = generate_score(df_returns, dis_res, method, lookback_for_factors=252, lookback_for_residual=60, export=True)
    #     dis_res = [dis_res[i * 769:769 * (i+1)] for i in range(4)]
    #     processes = [mp.Process(target=generate_score, args=(i, j, method, 252, 60, args.save_outputs)) for i,j in zip(df, dis_res)]
    #     for p in processes:
    #         p.start()
    #     for p in processes:
    #         p.join()
    #
    #     time.sleep(1)
    #     pidnums = [int(x) for x in os.listdir('tmp')]
    #     pidnums.sort()
    #     if args.gas:
    #         file_list = ['df_score_gas', 'dis_res_reg_gas', 'b_values_gas']
    #     else:
    #         file_list = ['df_score', 'dis_res_reg', 'b_values', 'R_squared']
    #     logging.info('Merging files...')
    #     file_merge(pidnums, file_list)
    #     for x in os.listdir('tmp'):
    #         os.remove(f'tmp/{x}')
    #
    # # df_score, beta_tensor = generate_score(df_returns, n_factor=args.n_components, variable_n_factor=args.variable_number, lookback_for_factors=252, lookback_for_residual=60, export=args.save_outputs)
    #
    # # spy = pd.read_csv(go_up(1) + '/saved_data/spy.csv')
    # # beta_spy = SPY_beta(df_returns, spy, export=args.save_outputs)

    time_elapsed = (end - start)
    logging.info('Time required for generate s-scores: %.2f seconds' %time_elapsed)
