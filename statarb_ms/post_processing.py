import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from makedir import go_up
import argparse
import logging

def plot_returns(name, density):
    trading_days = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date
    x_label_position = np.arange(0, len(trading_days)-252, density)
    x_label_day = [trading_days[i] for i in x_label_position]
    ret = np.load(go_up(1) + f'/saved_data/{name}.npy')
    plt.plot(ret, 'k')
    plt.xticks(x_label_position, x_label_day, rotation=60)
    plt.grid()
    plt.show()

def file_merge(*pidnums):
    df_score = [pd.read_csv(go_up(1) + f'/saved_data/df_score_{i}.csv') for i in pidnums]
    beta_tensor = [np.load(go_up(1) + f'/saved_data/beta_tensor_{i}.npy') for i in pidnums]
    Q = [np.load(go_up(1) + f'/saved_data/Q_{i}.npy') for i in pidnums]
    b_values = [np.load(go_up(1) + f'/saved_data/b_values_{i}.npy') for i in pidnums]
    R_squared = [np.load(go_up(1) + f'/saved_data/R_squared_{i}.npy') for i in pidnums]

    pd.concat(df_score, ignore_index=True).to_csv(go_up(1) + '/saved_data/ScoreData.csv', index=False)
    np.save(go_up(1) + '/saved_data/beta_tensor', np.vstack(beta_tensor))
    np.save(go_up(1) + '/saved_data/Q', np.vstack(Q))
    np.save(go_up(1) + '/saved_data/b_values', np.vstack(b_values))
    np.save(go_up(1) + '/saved_data/R_squared', np.vstack(R_squared))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generator of historical price data')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument("-p", "--plots", action='store_true',
                        help=("Plot returns"))
    parser.add_argument('-m', '--merge', action='store_true',
                        help='Merge files outputed by scoring.py')
    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    if args.plots:
        name = input('Name of the returns file: ')
        density = int(input('Density of the labels on x-axis: '))
        plot_returns(name, density)

    if args.merge:
        logging.warning('Assuming number of processes used is 5...')
        num_pid = int(input('First pid number: '))
        file_merge(num_pid, num_pid+1, num_pid+2, num_pid+3, num_pid+4)




# s_bo, s_so, s_bc, s_sc = -1.25, 1.25, 0.75, -0.50
# score = pd.read_csv(go_up(1) + '/saved_data/ScoreData.csv').AAPL
# plt.plot(score,'k', alpha=0.8, linewidth=0.5)
# plt.xticks(x_label_position, x_label_day, rotation=60)
# plt.hlines(s_bo, 0, score.shape[0], color='green', linestyle='dashed',label='Open long trade')
# plt.hlines(s_sc, 0, score.shape[0], color='red', linestyle='dashed', label='Close long trade')
# plt.hlines(s_so, 0, score.shape[0], color='red', linestyle='dashed', label='Open short trade')
# plt.hlines(s_bc, 0, score.shape[0], color='green', linestyle='dashed', label='Close short trade')
# plt.legend()
# plt.show()
