import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from makedir import go_up
from tqdm import tqdm


def pca(df_returns, n_components, variable_number=False, threshold=0.55):
    '''
    Compute the PCA decomposition of a dataset.
    Parameters
    ----------
    df_returns_price : str
        Path to the csv dataframe file.
    n_components : int
        Number of components you want your dataframe be projected to.
    variable_n_factor : bool
        to be set to True if the number of principal components is chosen through a fixed amount of explained variance, False if it is set by n_components.
        If it is True, n_components can be any value: the PCA will ignore and not use it.
    threshold : float
        Explained variance to keep. This parameter acts only when variable_n_factor is set to True.
    Returns
    -------
    eigenvalues, eigenvectors : list
    '''

    scaler = StandardScaler()
    df_returns_norm = pd.DataFrame(scaler.fit_transform(
        df_returns), columns=df_returns.columns)
    if variable_number:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    pca.fit(df_returns_norm.values)
    n_pca = pca.n_components_
    eigenvalues = pca.explained_variance_ratio_
    eigenvectors = pca.components_

    if variable_number:
        explained_variance = 0
        for i, eigenval in zip(range(len(eigenvalues)), pca.explained_variance_ratio_):
            explained_variance += eigenval
            if explained_variance >= threshold:
                eigenvalues, eigenvectors = pca.explained_variance_ratio_[
                    :i], pca.components_[:i]
                break

    logging.info(
        f"Fraction of variance preserved: {eigenvalues.sum():.2f}")

    return eigenvalues, eigenvectors


def eigenportfolios(df_returns, eigenvectors):
    '''
    Ordering of the components of the eigenvectors of the correlation matrix (principal components).
    Example: the first principal component is v1 = (v11, v12, ... , v1N). Each component represent one company.
    The first dictionary is then constructed ordering the vij and associating to each of them the corresponding
    company.

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
    ''' This function will output a matrix of shape n_factors x n_stocks in which the i-j elements corresponds
    to the amount of money to be invested in the company j relative to the risk_factor i
    '''
    dev_t = df_returns.std()
    q = np.zeros(shape=(eigenvectors.shape[0], df_returns.shape[1]))
    for i in range(eigenvectors.shape[0]):
        q[i] = eigenvectors[i,:] * dev_t

    return q


def risk_factors(df_returns, eigenvectors, export=False):
    '''This function evaluates the returns of the eigenportfolios. In the PCA
    approach the returns of the eigenportolios are the factors

    Parameters
    ----------
    df_returns : pandas.core.frame.Dataframe
        Dataframe of 1 day returns for each stock.
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
    factors = np.zeros((n_days, n_factors))
    dev_t = 1 / df_returns.std()
    logging.info(f'Number of factors extracted from PCA: {n_factors}')
    returns = df_returns.values

    for j in range(n_factors):
        for i in range(n_days):
            factors[i, j] = (returns[i] *
                             dev_t * eigenvectors[j]).sum()

    if export:
        name = input('Name of the file that will be saved: ')
        np.save(go_up(1) + f'/saved_data/{name}', factors)

    return np.array(factors)


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

    df_returns = pd.read_csv(go_up(1) + "/saved_data/ReturnsData.csv")
    eigenvalues, eigenvectors = pca(df_returns, n_components=args.n_components,
                                    variable_number=args.variable_number, threshold=args.threshold_variance)
    # factors = risk_factors(df_returns, eigenvectors, export=args.export)
    eigenportfolio = eigenportfolios(df_returns, eigenvectors)
    q = money_on_stock(df_returns, eigenvectors)

    if args.plots:
        plt.figure()
        plt.bar(np.arange(1, len(eigenvalues) + 1), eigenvalues, alpha=0.8)
        plt.xlabel('Eigenvalues')
        plt.ylabel('Variance Explained')
        plt.show()

        plt.figure()
        plt.hist(eigenvalues, bins=30, alpha=0.8)
        plt.title('Density of states')
        plt.show()

        # explained_variance = []
        # for i in range(0,6301,2):
        #     eigenvalues, eigenvectors = pca(df_returns[i:i+look_back], n_components=args.n_components,
        #                                     variable_number=args.variable_number, threshold=args.threshold_variance)
        #     explained_variance.append(eigenvalues.sum())
        # plt.figure()
        # plt.plot(list(range(0,6301,2)), explained_variance)
        # plt.title(f'Explained variance in time with {args.n_components} PCA components')
        # plt.xlabel('Time')
        # plt.ylabel('Explained variance')
        # plt.show()
