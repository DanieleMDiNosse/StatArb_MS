import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from factors import pca, risk_factors
from sklearn import linear_model
from statsmodels.tsa.ar_model import AutoReg


def regression(X, Y, const=True):
    '''
    Simple linear regression model that relies on StatsModels OLS

    Parameters
    ----------
    X : numpy ndarray
        Features.
    Y : numpy ndarray
        Target.
    '''

    if const:
        X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    res = model.fit()
    beta0 = np.array(res.params[0])
    betas = np.array(res.params[1:])
    conf_intervals = np.array(res.conf_int(alpha=0.05, cols=None))
    predictions = model.predict(res.params)
    residuals = np.array(res.resid)
    rsquared = res.rsquared

    return beta0, betas, conf_intervals, residuals, predictions, rsquared


def auto_regression(X):
    mod = AutoReg(X, lags=1, old_names=False)
    res = mod.fit()
    par = res.params
    pred = mod.predict(res.params)
    resid = np.array(res.resid)
    conf_int = np.array(res.conf_int(alpha=0.05, cols=None))

    return par, pred, resid, conf_int


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-vn", "--variable_number", action='store_true',
                        help=("Use a variable number of PCA components. Each time the explained variance is 0.55. The default is False"))
    parser.add_argument('-n', '--n_components', type=int, default=15,
                        help='Number of PCA components to keep. Valid if variable_number is False. The default is 15')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    df_returns = pd.read_csv(
        "/home/danielemdn/Documents/thesis/StatArb_MS/saved_data/ReturnsData.csv")
    # eigenvalues, eigenvectors = pca(
    #     df_returns[:252], n_components=args.n_components, variable_number=args.variable_number)
    # factors = risk_factors(df_returns, eigenvectors)

    # beta0, betas, residuals, conf_intervals = regression(factors, df_returns.AN)
    a, b, predictions, residuals, conf_intervals = auto_regression(
        df_returns.AN)
    print(a, b)
    plt.figure()
    plt.hist(residuals, bins=500)
    plt.show()

    # plt.figure()
    # plt.plot(df_returns.AAPL-1, label='Apple')
    # plt.plot(predictions, label='Linear prediction')
    # plt.legend()
    # plt.grid()
    # plt.show()
