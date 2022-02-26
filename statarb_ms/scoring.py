import numpy as np
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from makedir import go_up
from post_processing import file_merge, remove_file
from reduced_loglikelihood import complete_loglikelihood, loglikelihood, targeting_loglikelihood
from gas import estimation
from factors import pca, risk_factors, money_on_stock
from regression_parameters import regression, auto_regression
from sklearn.metrics import r2_score
import time
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
from scipy.stats import normaltest


def generate_data(df_returns, n_factor, method, targeting_estimation, lookback_for_factors=252, lookback_for_residual=60, export=True):
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
        Weights of each of the pca components for each day and each stock. Dimensions are (trading days x n_stocks x n_factors).
    Q : numpy ndarray
        Money to invest on each factors for each days. Dimensions are dim (trading days x n_factors x n_components x n_stocks)
    '''

    trading_days = df_returns.shape[0] - lookback_for_factors # 6294
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    # res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    score = np.zeros(shape=(trading_days, n_stocks)) # Il primo score corrisponderà al 252° giorno (indice 251)
    b_values = np.zeros(shape=(trading_days, n_stocks, 3))
    b_values_gas = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    a_values_gas = np.zeros(shape=(trading_days, n_stocks, 1))
    R_squared = np.zeros(shape=(trading_days, n_stocks))
    dis_res_reg = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    c = 0 # counter to track the number of negative bs comes from GAS model

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(trading_days)):
        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        period = df_returns[i:lookback_for_factors + i] # [0,252[, [1,253[ ecc -> ogni period comprende un anno di trading (252 giorni)
        eigenvalues, eigenvectors = pca(period, n_components=n_factor)
        Q[i,:,:] = money_on_stock(period, eigenvectors)# trading_days x n_factors x n_stocks. Ogni giorno so quanto investire su ogni compagnia all'interno di ognuno dei fattori
        factors = risk_factors(period, Q[i], eigenvectors)# ritorni dei fattori di rischio per ogni periodo
        # Ottenuti i fattori di rischio si procede con la stima del processo dei residui per ogni compagnia.
        for stock in df_returns.columns:
            stock_idx = df_returns.columns.get_loc(stock)
            beta0, betas, conf_inter, residuals, pred, _ = regression(factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])
            beta_tensor[i, stock_idx, :] = betas
            # res[i, stock_idx, :] = residuals
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
                init_params = np.random.uniform(0, 1, size=4)
                b, a, xi, _ = estimation(complete_loglikelihood, discreteOU, init_params, method='Nelder-Mead', targeting_estimation=False)
                b_values_gas[i, stock_idx, :] = b
                a_values_gas[i, stock_idx, :] = a
                b = b[-1] # Mi serve solo l'ultimo valore per lo score
                if b < 0: # se b è negativo, sostituiscilo con quello stimato supponendo sia costante nella finestra temporale
                    discreteOU = np.append(discreteOU, discreteOU[-1])
                    parameters, discrete_pred, discrete_resid, discrete_conf_int = auto_regression(discreteOU)
                    a, b = parameters[0], parameters[1]
                    c += 1

            if b == 0.0:
                print(f'B NULLO PER {stock}')
                break

            k = -np.log(b) * lookback_for_factors
            if k < lookback_for_factors / (0.5 * lookback_for_residual):
                score[i, stock_idx] = 0
            else:
                m = a / (1 - b)
                if method == 'constant_speed': sgm_eq = np.std(discrete_resid) * np.sqrt(1 / (1 - b * b))
                if method == 'gas_modelization': sgm_eq = np.std(xi) * np.sqrt(1 / (1 - b * b))
                # naive method. Keep in mind that s-score depends on the risk factors
                score[i, stock_idx] = -m / sgm_eq
                print(i, stock, score[i, stock_idx])
                time.sleep(2)

    with open(go_up(1) + f'/saved_data/negative_b_{os.getpid()}', 'w', encoding='utf-8') as file:
        file.write(f'Number of b values for process {os.getpid()}: {c}')

    df_score = pd.DataFrame(score, columns=df_returns.columns)
    if export:
        if method == 'gas_modelization':
            df_score.to_csv(go_up(1) + f'/saved_data/df_score_gas_{os.getpid()}.csv', index=False)
            np.save(go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
            np.save(go_up(1) + f'/saved_data/b_gas_{os.getpid()}', b_values_gas)
            np.save(go_up(1) + f'/saved_data/a_gas_{os.getpid()}', a_values_gas)
            np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)
            np.save(go_up(1) + f'/saved_data/dis_res_{os.getpid()}', dis_res)
            # np.save(go_up(1) + f'/saved_data/res_{os.getpid()}', res)

        if method == 'constant_speed':
            df_score.to_csv(go_up(1) + f'/saved_data/df_score_{os.getpid()}.csv', index=False)
            # np.save(go_up(1) + f'/saved_data/dis_res_reg_{os.getpid()}', dis_res_reg)
            np.save(go_up(1) + f'/saved_data/b_values_{os.getpid()}', b_values)
            # np.save(go_up(1) + f'/saved_data/R_squared_{os.getpid()}', R_squared)
            np.save(go_up(1) + f'/saved_data/beta_tensor_{os.getpid()}', beta_tensor)
            np.save(go_up(1) + f'/saved_data/Q_{os.getpid()}', Q)
            # np.save(go_up(1) + f'/saved_data/dis_res_{os.getpid()}', dis_res)

def SPY_beta(df_returns, spy, lookback_for_factors=252, lookback_for_residual=60, export=True):

    df_returns = np.array(df_returns)
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    vec_beta_spy = np.zeros(shape=(trading_days, n_stocks))
    for i in tqdm(range(trading_days), desc='SPY'):
        period = df_returns[i:lookback_for_factors + i]
        spy_period = spy[i:lookback_for_factors + i]

        for stock in range(df_returns.shape[1]):
            beta0, betas, conf_intervals, residuals, predictions, rsquared = regression(
                spy_period[-lookback_for_residual:], period[-lookback_for_residual:, stock])
            vec_beta_spy[i, stock] = betas[0]
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
    parser.add_argument("-sp", "--spy", action='store_true',
                        help=("Scoring for SPY."))
    parser.add_argument('-r', '--range', action='store_true',
                        help='Select a specific time range between 1995-01-03 to 2020-12-31')
    parser.add_argument('-s', '--save_outputs', action='store_false',
                        help='Choose whether or not to save the outputs. The default is True')
    parser.add_argument("-t", "--targ_est", action='store_true')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    start = time.time()

    df_returns = pd.read_csv(go_up(1) +
                             "/saved_data/ReturnsData.csv")[:-300]
    # df_returns = pd.DataFrame(df_returns[:-300].values, columns=df_returns.columns.to_list())

    if args.spy:
        spy = pd.read_csv(go_up(1) + "/saved_data/spy.csv")[2770:6042]
        SPY_beta(df_returns, spy)

    # df = [df_returns[:1510], df_returns[1258:2769], df_returns[2517:4028], df_returns[3776:5287], df_returns[5035:]]
    df = [df_returns[:1000], df_returns[748:1749], df_returns[1497:2498], df_returns[2246:3247], df_returns[2995:3996], df_returns[3744:4745], df_returns[4493:5494],df_returns[5242:]]
    # df = [df_returns[:800], df_returns[548:1349], df_returns[1097:1898], df_returns[1646:2447], df_returns[2195:2996], df_returns[2774:]]


    if args.gas == True: method = 'gas_modelization'
    else: method = 'constant_speed'

    if args.range:
        start =str(input('Start date (YYY-MM-DD): '))
        end = str(input('End date (YYY-MM-DD): '))
        date = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv')
        start = date.index[date.Date == start].tolist()[0]
        end = date.index[date.Date == end].tolist()[0]
        print(start, end)
        df_returns = pd.read_csv(go_up(1) +
                                 "/saved_data/ReturnsData.csv")[start: end + 1]
        generate_data(df_returns, args.n_components, method)
        end = time.time()

    else:
        processes = [mp.Process(target=generate_data, args=(i, args.n_components, method, args.targ_est, 252, 60, args.save_outputs)) for i in df]
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
            # file_list = ['dis_res']
            file_list = ['df_score_gas', 'beta_tensor', 'b_gas', 'Q', 'a_gas', 'dis_res']
        else:
            # file_list = ['beta_tensor', 'Q', 'dis_res', 'df_score', 'dis_res_reg', 'b_values', 'R_squared']
            file_list = ['beta_tensor', 'Q', 'df_score', 'b_values']
        logging.info('Merging files...')
        file_merge(pidnums, file_list)
        remove_file(pidnums, file_list)
        os.system('rm tmp/*')

    time_elapsed = (end - start)
    logging.info('Time required for generate s-scores: %.2f seconds' %time_elapsed)
