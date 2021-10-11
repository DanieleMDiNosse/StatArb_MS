from sklearn import linear_model
from factors import pca, risk_factors
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd


def regression(x, y):
    '''
    Simple linear regression model that relies on sklearn LinearRegression
    '''
    lm = linear_model.LinearRegression(fit_intercept=False)
    model = lm.fit(x, y)
    r2 = lm.score(x, y)
    predictions = lm.predict(x)
    residuals = np.array(y) - np.array(predictions)
    betas = lm.coef_
    beta0 = lm.intercept_

    return predictions, beta0, betas, residuals


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
        "/home/danielemdn/Documents/thesis/StatArb_MS/ReturnsData.csv")
    eigenvalues, eigenvectors = pca(
        df_returns, n_components=args.n_components, variable_number=args.variable_number)
    factors = risk_factors(df_returns, eigenvectors)

    # Regression of the stock returns on the PCA components
    predictions, beta0, betas, residuals = regression(factors, df_returns.AN)

    discreteOU = np.zeros(len(residuals[-60:]))
    for i in range(len(discreteOU)):
        discreteOU[i] = residuals[:i + 1].sum()
    discreteOU_predicted, a, b, residuals = regression(
        discreteOU[:-1].reshape(-1, 1), discreteOU[1:])  # NON VA BENE

    # plt.figure()
    # plt.plot(df_returns.AAPL-1, label='Apple')
    # plt.plot(predictions, label='Linear prediction')
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure()
    plt.plot(discreteOU, label='discreteOU')
    plt.plot(discreteOU_predicted, label='discreteOU_predicted')
    plt.legend()
    plt.grid()
    plt.show()
