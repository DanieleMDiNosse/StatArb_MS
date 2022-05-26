import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from makedir import go_up
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import seaborn as sns


def pca(df_returns, n_components, verbose=False):
    '''
    Compute the PCA decomposition of a dataset. This function relies on PCA module in sklearn.decomposition package.

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Dataframe of 1-day returns for each stock.
    n_components : int, bool
        Number of components you want your dataframe be projected into.
    verbose : bool (optional)
        If True, print information about variance explained by the n_components.

    Returns
    -------
    eigenvectors : ndarray of shape (n_components, n_stocks)
        Principal components.
    eigenvalues : ndarray of shape (n_components)
        Eigenvalues of the empirical correlation matrix. They represent the explained variance along each principal components.
    '''

    scaler = StandardScaler()
    df_returns_norm = pd.DataFrame(scaler.fit_transform(
        df_returns), columns=df_returns.columns)
    if n_components == 0:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    pca.fit(df_returns_norm.values)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    explained_variance = pca.explained_variance_ratio_

    if verbose:
        logging.info(
            f"Fraction of variance preserved with {n_components} components: {explained_variance.sum()}")

    return eigenvalues, -eigenvectors, explained_variance.sum()


def eigenportfolios(df_returns, eigenvectors):
    '''
    Ordering of the components of the eigenvectors of the correlation matrix (principal components).
    Example: the first principal component is v1 = (v11, v12, ... , v1N). Each component vij represent the projection of the i-th stock on the j-th principal component.
    This function creates a dictionary constructed ordering the vij and associating to each of them the corresponding company tick.

    Parameters
    ----------
    df_returns : pandas.core.frame.DataFrame
        Pandas dataframe of 1-day returns
    eigenvectors : numpy array
        Numpy array corresponding to the eigenvectors of the correlation matrix (principal components)

    Returns
    -------
    eigenporfolios : list
        List of dictionaries ordered according to the greatest variance explained by the principal components.
'''

    portfolio = [{df_returns.columns[i]: eigenvectors[j][i]
                  for i in range(df_returns.shape[1])} for j in range(len(eigenvectors))]
    eigenport = [dict(sorted(portfolio[i].items(
    ), key=lambda item: item[1])) for i in range(len(portfolio))]

    return eigenport


def money_on_stock(df_returns, eigenvectors):
    ''' This function will output a matrix of shape n_factors x n_stocks in which the (i,j) element corresponds to the weight of company j relative to the risk factor i.

    Parameters
    ----------
    df_returns : pandas.core.frame.Dataframe
        Dataframe of 1 day returns for each stock.
    eigenvectors : ndarray of shape (number of PCA components, number of stocks)
        Eigenvectors obtained from PCA.

    Returns
    -------
    q : numpy.ndarray of shape (n_factors, n_stocks)
        Weights of each company on each eigenvectors.
    '''

    dev_t = 1 / df_returns.std(axis=0)
    n_fact = eigenvectors.shape[0]
    n_stock = df_returns.shape[1]
    q = np.zeros(shape=(n_fact, n_stock))
    for i in range(n_fact):
        q[i] = eigenvectors[i, :] * dev_t

    return q


def risk_factors(df_returns, Q, eigenvectors, export=False):
    '''This function evaluates the returns of the eigenportfolios.

    Parameters
    ----------
    df_returns : pandas.core.frame.Dataframe
        Dataframe of 1 day returns for each stock.
    Q : ndarray
        Weights of each company on each eigenvectors.
    eigenvectors : numpy array
        Eigenvectors obtained from PCA.
    export : bool
        Choose whether or not to export to .npy the vector of eigenportfolios.

    Returns
    -------
    factors : numpy array
        Numpy array of eigenportfolios'''

    n_days = df_returns.shape[0]
    n_stocks = df_returns.shape[1]
    n_factors = len(eigenvectors)
    factors = np.zeros(shape=(n_days, n_factors))
    dev_t = 1 / df_returns.std()
    returns = df_returns.values

    for i in range(n_days):
        factors[i] = np.matmul(Q, df_returns.iloc[i])

    if export:
        name = input('Name of the factors file that will be saved: ')
        np.save(go_up(1) + f'/saved_data/{name}', factors)

    return factors


def theoretical_eigenvalue_distribution(N, T, var, eigenvalues):
    ''' Following the results of Random Matrix Theory, this function returns the eigenvalues theoretical distribution of a purely random matrix with T rows and N columns.

    Parameters
    ----------
    N : int
        Number of features (columns)
    T : int
        Number of rows.
    var : float
        Variance of the elements of the matrix.
    eigenvalues : ndarray of shape (N)
        Eigenvalues of the sample correlation matrix.

    Returns
    -------
    rho : ndarray of shape (N)
        Theoretical distribution of the eigenvalues under the assumption that the matrix is random.
    '''

    Q = T / N
    lambda_max = var * (1 + 1 / Q + 2 * np.sqrt(1 / Q))
    lambda_min = var * (1 + 1 / Q - 2 * np.sqrt(1 / Q))
    rho = np.array([Q / (2 * np.pi * var) * np.sqrt((lambda_max - lam)
                   * (lam - lambda_min)) / lam for lam in eigenvalues])

    return rho


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-vn", "--variable_number", action='store_true',
                        help=("Use a variable number of PCA components. Each time the explained variance is 0.55. The default is False"))
    parser.add_argument('-n', '--n_components', type=int,
                        help='Number of PCA components to keep. Valid if variable_number is False.')
    parser.add_argument('-tv', '--threshold_variance', type=float, default=0.55,
                        help='Explained variance to keep. Valid if variable_number is True. The default is 0.55')
    parser.add_argument('-p', '--plots', action='store_true',
                        help='Display some plots')
    parser.add_argument('-e', '--export', action='store_false',
                        help='Choose whether or not to save factors as npy. The default is False')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])
    plt.style.use('seaborn')

    df_returns = pd.read_pickle("/mnt/saved_data/ReturnsData.pkl")
    eigenval, eigenvec, expvar = pca(df_returns[:252], n_components=15)
    Q = money_on_stock(df_returns[:252], eigenvec)
    factors = risk_factors(df_returns[:252], Q, eigenvec, export=False)
    N = df_returns.shape[1]
    days = df_returns.shape[0]
    T = 252


    columns = [fr'${i}^°$' for i in range(1,16)]
    df_eigenvec = pd.DataFrame(factors, columns=columns)
    print(df_eigenvec.shape)
    sns.heatmap(df_eigenvec.corr())
    plt.show()
    # var = np.zeros(shape=(days - T))
    # first_var = np.zeros(shape=(days - T))
    # for day in tqdm(range(days-T)):
    #     eigenvalues, eigenvectors, explained_variance = pca(df_returns[day: day + T], n_components=15)
    #     var[day] = explained_variance
    #     eigenvalues, eigenvectors, explained_variance = pca(df_returns[day: day + T], n_components=1)
    #     first_var[day] = explained_variance
    # plt.figure(figsize=(12,8), tight_layout=True)
    # plt.plot(var, 'k', linewidth=1.3, alpha=0.75, label=r'$\Sigma_r$ for 15 PCs')
    # plt.plot(first_var, 'b', linewidth=1.3, alpha=0.75, label=r'$\Sigma_r$ for 1 PCs')
    # plt.legend(prop={'size': 13})
    # trading_days = pd.read_pickle('/mnt/saved_data/PriceData.pkl').Date
    # x_label_position = np.arange(0, len(trading_days)-T, T)
    # x_label_day = [trading_days[i] for i in x_label_position]
    # plt.xticks(x_label_position, x_label_day, rotation=90)
    # plt.show()

    # plt.plot(np.convolve(eigenvalues, window * 1. / sum(window), mode='same'))
    # eigenportfolio = eigenportfolios(df_returns, eigenvectors)
    # first_components = list(eigenportfolio[0].values())
    # first_ticks = list(eigenportfolio[0].keys())
    # print(first_ticks)
    # plt.scatter(first_ticks, first_components, s=1.5)
    # plt.show()
    if args.plots:
        trading_days = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date
        x_label_position = np.arange(0, len(trading_days)-252, 252)
        x_label_day = [trading_days[i] for i in x_label_position]
        plt.figure(figsize=(12,5), tight_layout=True)
        # plt.bar(np.arange(len(eigenvalues)), eigenvalues,color='k', alpha=0.8)
        # plt.xlabel('Eigenvalues')
        # plt.ylabel('Explained Variance')
        # plt.show()

        plt.hist(eigenvalues, color='k', density=True, bins=300, alpha=0.8)
        # sns.displot(np.array(eigenvalues), kind='kde', bw_adjust=0.1)
        plt.plot(eigenvalues, theoretical_eigenvalue_distribution(
            N, T, 0.7, eigenvalues), 'crimson', linewidth=2, label=r'$\sigma^2 = 0.7$')
        plt.title('Density of states')
        # plt.legend()
        plt.show()

        # explained_variance = []
        # look_back = 252
        # for i in tqdm(range(0,6301,2)):
        #     eigenvalues, eigenvectors = pca(df_returns[i:i+look_back], n_components=args.n_components,
        #                                     variable_number=args.variable_number, threshold=args.threshold_variance)
        #     explained_variance.append(eigenvalues.sum())
        # plt.figure()
        # plt.plot(list(range(0,6301,2)), explained_variance, 'k')
        # plt.ylabel('Explained variance')
        # plt.xticks(x_label_position, x_label_day, rotation=60)
        # plt.grid(True)
        # plt.show()
