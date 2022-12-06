# import numpy as np
import argparse
import logging
import multiprocessing as mp
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import telegram_send
from data import price_data
from factors import money_on_stock, pca, risk_factors
from gas import estimation
from loglikelihood import loglikelihood
from post_processing import LM_test_statistic, file_merge
from regression_parameters import auto_regression, regression
from scipy.stats import normaltest
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def generate_data(df_returns, n_factor=15, lookback_for_factors=252, lookback_for_residual=60, export=True, path='/mnt/hdd/saved_data'):
    '''This function uses an amount of days equal to lookback_for_factors to evaluate the PCA components and then uses them as regressor
    for stock returns. Once terminated, the function saves the parameters of the regressions (alphas and beta_tensor) and the residuals (res)
    together with its cumulative sum (dis_res).

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns for each stock.
    n_factor : int
        Number of principal components to keep in the PCA algorithm. The default is 15.
    lookback_for_factors : int
        Number of days used for the PCA. The default is 252 (one year of trade).
    lookback_for_residual : inf
        Number of days used for the estimation of the residual process (OU). The default is 60.
    export : bool
        Choose whether or not to export to csv the s_score dataframe and to .npy the vector of betas.

    Returns
    -------
    None
    '''

    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    beta_tensor = np.zeros(shape=(trading_days, n_stocks, n_factor))
    alphas = np.zeros(shape=(trading_days, n_stocks))
    Q = np.zeros(shape=(trading_days, n_factor, n_stocks))
    dis_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))

    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(2), desc=f'{os.getpid()}'):
        # Una finestra temporale di 252 giorni è traslata di 1 giorno. Ogni volta viene eseguita una PCA su tale periodo
        # ed i fattori di rischio sono quindi valutati.
        # [0,252[, [1,253[ ecc -> ogni period comprende un anno di trading (252 giorni)
        period = df_returns[i:lookback_for_factors + i]
        eigenvalues, eigenvectors, exp_var = pca(period, n_components=n_factor)
        # trading_days x n_factors x n_stocks. Ogni giorno so quanto investire su ogni compagnia all'interno di ognuno dei fattori
        Q[i, :, :] = money_on_stock(period, eigenvectors)
        # ritorni dei fattori di rischio per ogni periodo
        factors = risk_factors(period, Q[i], eigenvectors)
        # Ottenuti i fattori di rischio si procede con la stima del processo dei residui per ogni compagnia.
        for stock in df_returns.columns[:2]:

            stock_idx = df_returns.columns.get_loc(stock)
            beta0, betas, conf_inter, residuals, pred, _ = regression(
                factors[-lookback_for_residual:], period[-lookback_for_residual:][stock])

            # Avoid estimation of non stationary residuals
            p_value = adfuller(residuals)[1]
            if p_value > 0.05:
                continue

            alphas[i, stock_idx] = beta0 * lookback_for_factors
            beta_tensor[i, stock_idx, :] = betas
            res[i, stock_idx, :] = residuals
            X = np.cumsum(residuals)
            dis_res[i, stock_idx, :] = X

    if export:
        np.save(
            f'{path}/beta_tensor_{os.getpid()}', beta_tensor)
        np.save(f'{path}/alphas_{os.getpid()}', alphas)
        np.save(f'{path}/Q_{os.getpid()}', Q)
        np.save(f'{path}/dis_res_{os.getpid()}', dis_res)
        np.save(f'{path}/res_{os.getpid()}', res)


def only_scoring(dis_res, df_returns, method, link_fun, n_iter, lookback_for_factors, lookback_for_residual, path):
    ''' This function uses an amount equal to lookback_for_residual to evaluate the parameters of the Ornstein-Uhlenbeck process for the residuals
     and then compute the s-scores.
    '''

    num_par = 4
    targeting_estimation = False
    trading_days = df_returns.shape[0] - lookback_for_factors
    n_stocks = df_returns.shape[1]
    score = np.zeros(shape=(trading_days, n_stocks))
    estimates = np.zeros(shape=(trading_days, n_stocks, num_par + 1))
    bs = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    r2 = np.zeros(shape=(trading_days, n_stocks))
    const_AR_par = np.zeros(shape=(trading_days, n_stocks, 2))
    AR_res = np.zeros(shape=(trading_days, n_stocks, lookback_for_residual))
    AR_res_gas = np.zeros(
        shape=(trading_days, n_stocks, lookback_for_residual))
    sgm_eq = np.zeros(shape=(trading_days, n_stocks))
    kappas = np.zeros(shape=(trading_days, n_stocks))

    # I initialize some counters.
    c, cc, ccc = 0, 0, 0

    # I create some empty file named as the process PID, in order to use them to merge all the files (via file_merge function) that the different processes will create.
    with open(f'tmp/{os.getpid()}', 'w', encoding='utf-8') as file:
        pass

    for i in tqdm(range(2), desc=f'{os.getpid()}'):
        for stock in df_returns.columns[:1]:
            stock_idx = df_returns.columns.get_loc(stock)
            X = dis_res[i, stock_idx, :]

            # Avoid estimation on zero residuals
            if X.any(axis=0) == 0:
                cc += 1
                continue

            if method == 'constant_speed':
                X = np.append(X, X[-1])
                parameters, discrete_pred, xi, discrete_conf_int = auto_regression(
                    X)
                a, b = parameters[0], parameters[1]
                # r2[i, stock_idx] = r2_score(X, np.array(discrete_pred))
                const_AR_par[i, stock_idx, :] = parameters
                AR_res[i, stock_idx, :] = xi

            if method == 'gas_modelization':
                if targeting_estimation:
                    b, a, xi, est, std = estimation(
                        X, n_iter, targeting_estimation=targeting_estimation)
                else:
                    b, a, xi, est = estimation(X, n_iter, link_fun)
                AR_res_gas[i, stock_idx, :] = xi
                estimates[i, stock_idx, :-1] = est
                estimates[i, stock_idx, -1] = 1  # sigma = 1 by assumption
                bs[i, stock_idx, :] = b
                b = b[-1]

                if b < 0:
                    X = np.append(X, X[-1])
                    parameters, discrete_pred, xi, discrete_conf_int = auto_regression(
                        X)
                    a, b = parameters[0], parameters[1]
                    c += 1

            if b == 0.0:
                cc += 1
                continue
            else:
                k = -np.log(b) * lookback_for_factors
                kappas[i, stock_idx] = k
                if k < lookback_for_factors / (0.5 * lookback_for_residual):
                    score[i, stock_idx] = 0
                    ccc += 1
                else:
                    m = a / (1 - b)
                    sgm_eq[i, stock_idx] = np.std(xi) * np.sqrt(1 / (1 - b**2))
                    score[i, stock_idx] = -m / sgm_eq[i, stock_idx]

    # logging.info(f'Total number of estimation for process {os.getpid()}: {n_stocks * trading_days}', f'Number of negative b values for process {os.getpid()}: {c}',
    #              f'Number of stock with speed of mean reversion refused for process {os.getpid()}: {ccc}', f'Number of zero b for process {os.getpid()}: {cc}')

    df_score = pd.DataFrame(score, columns=df_returns.columns)

    if method == 'gas_modelization':
        df_score.to_pickle(f'{path}/df_score_gas_{os.getpid()}.pkl')
        np.save(f'{path}/estimates_gas_{os.getpid()}', estimates)
        np.save(f'{path}/bs_{os.getpid()}', bs)
        np.save(f'{path}/sgm_eq_gas_{os.getpid()}', sgm_eq)
        np.save(f'{path}/kappas_gas_{os.getpid()}', kappas)
        np.save(f'{path}/AR_res_gas_{os.getpid()}', AR_res_gas)

    if method == 'constant_speed':
        df_score.to_pickle(f'{path}/df_score_{os.getpid()}.pkl')
        np.save(f'{path}/r2_{os.getpid()}', r2)
        np.save(f'{path}/const_AR_par_{os.getpid()}', const_AR_par)
        np.save(f'{path}/sgm_eq_{os.getpid()}', sgm_eq)
        np.save(f'{path}/AR_res_{os.getpid()}', AR_res)
        np.save(f'{path}/kappas_{os.getpid()}', kappas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-s", "--scoring", action='store_true',
                        help='If passed, compute only the s-scores, provided residuals are already been computed.')
    parser.add_argument("-g", "--gas", action='store_true',
                        help=("Use gas estimation for the mean reverting speed. The default is False."))
    parser.add_argument('-lf', '--link_fun', type=int, help='In GAS filtering, choose what type of link function/distribution is adopted. 0 -> Identity link function with N(0,1) distribution; 1-> Logistic link function with N(0,1) distribution; 2-> Identity link function with Student t distribution.')
    parser.add_argument('-vi', '--vol_int', action='store_true',
                        help='Use return dataframe weighted  with volume information')
    parser.add_argument('-t', '--test_set', action='store_true',
                        help='Use test set with the hyperparameters tuned on the validation set')
    parser.add_argument("-nit", "--n_iter", type=int,
                        help='Number of Nelder-Mead optimization (minus 1) before the final BFGS optimization.', default=2)
    parser.add_argument("-sr", "--server", action="store_true", help="If passed, use paths for running the code on the SNS server")

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    plt.style.use('seaborn')
    warnings.filterwarnings('ignore')
    start = time.time()
    np.random.seed()

    if args.server:
        path = '/home/ddinosse/saved_data'
    else:
        path = '/mnt/hdd/saved_data'


# ---------- Volume integrated returns or simple returns ----------
    if args.vol_int:
        # path = input("Path of ReturnsVolData: ")
        logging.info(
            'Using volume integrated returns')
        if args.test_set:
            logging.info('Using test set')
            df_returns = pd.read_pickle(
                f"{path}/returns/ReturnsVolData.pkl")[4029:]
        else:
            logging.info('Using validation set')
            df_returns = pd.read_pickle(
                f"{path}/returns/ReturnsVolData.pkl")[:4030]
    else:
        # path = input("Path of ReturnsData: ")
        logging.info('Using the simple returns')
        if args.test_set:
            logging.info('Using test set')
            df_returns = pd.read_pickle(
                f"{path}/returns/ReturnsData.pkl")[4029:]
        else:
            logging.info('Using validation set')
            df_returns = pd.read_pickle(
                f"{path}/returns/ReturnsData.pkl")[:4030]
# ------------------------------------------------------------

# ---------- Split validation test or test test for multiprocessing ----------
    if args.test_set:
        df = [df_returns[:565], df_returns[313:879], df_returns[627:1193], df_returns[941:1507],
              df_returns[1255:1821], df_returns[1569:2135], df_returns[1883:2449], df_returns[2197:]]
    else:
        df = [df_returns[:750], df_returns[498:1249], df_returns[997:1748], df_returns[1496:2247],
              df_returns[1995:2746], df_returns[2494:3245], df_returns[2993:3744], df_returns[3492:]]
# ----------------------------------------------------------------------------


# ================ COMPUTE ONLY THE SCORES, provided that residuals are already computed ================
    if args.scoring:
        logging.info('Computing the s-scores...')
        # ---------- constant k or gas k ----------
        if args.gas == True:
            method = 'gas_modelization'
            logging.info('Estimation of k via GAS')

            if args.link_fun == 0:
                link_fun = 'identity'
            if args.link_fun == 1:
                link_fun = 'logistic'
            if args.link_fun == 2:
                link_fun = 'identity_student'
        else:
            method = 'constant_speed'
            link_fun = 0 # Anyway it is not used
            logging.info('Estimation of constant k')
        # ----------------------------------------
        # path = input("Path of the dis_res file: ")/home/ddinosse/
        length = int(input('Length of the estimation window for the scores: '))
        # name = input('Name of the dis_res file: ')
        logging.info(f'Length: {length}')
        dis_res = np.load(f"{path}/dis_res/dis_res{length}.npy")  # [:, stocks, :]

# ---------- Validation set or test set ----------
        if args.test_set:
            dr = [dis_res[:565, :, :], dis_res[313:879, :, :], dis_res[627:1193, :, :], dis_res[941:1507, :, :],
                  dis_res[1255:1821, :, :], dis_res[1569:2135, :, :], dis_res[1883:2449, :, :], dis_res[2197:, :, :]]
        else:
            dr = [dis_res[:750, :, :], dis_res[498:1249, :, :], dis_res[997:1748, :, :], dis_res[1496:2247, :, :],
                  dis_res[1995:2746, :, :], dis_res[2494:3245, :, :], dis_res[2993:3744, :, :], dis_res[3492:, :, :]]
# --------------------------------------------------

# ---------- Creation and initialization of the processes ----------
        processes = [mp.Process(target=only_scoring, args=(
            i, j, method, link_fun, args.n_iter, 252, length, path)) for i, j in zip(dr, df)]

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
            file_list = ['df_score_gas', 'estimates_gas',
                         'bs', 'sgm_eq_gas', 'kappas_gas', 'AR_res_gas']
        else:
            file_list = ['kappas', 'AR_res', 'sgm_eq',
                         'const_AR_par', 'r2', 'df_score']

        logging.info('Merging files, then remove splitted ones...')
        file_merge(pidnums, file_list, path)
        os.system('rm tmp/*')

# ==================== COMPUTE ONLY THE RESIDUALS ====================
    else:
        logging.info('Estimating residual process...')
        time.sleep(0.3)
        length = int(
            input('Lenght of the estimation window for the residuals: '))
        processes = [mp.Process(target=generate_data, args=(
            i, 15, 252, length, True, path)) for i in df]

        os.system('rm tmp/*')
        for p in processes:
            p.start()
            time.sleep(0.5)
        for p in processes:
            p.join()
        end = time.time()

        pidnums = [int(x) for x in os.listdir('tmp')]
        pidnums.sort()
        file_list = ['res', 'dis_res', 'beta_tensor', 'alphas', 'Q']

        logging.info('Merging files, then remove splitted ones...')
        file_merge(pidnums, file_list, path)
        os.system('rm tmp/*')

    time_elapsed = (end - start)
    logging.info('Time required: %.2f seconds' %
                 time_elapsed)
