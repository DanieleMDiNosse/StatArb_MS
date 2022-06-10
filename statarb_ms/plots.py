import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
plt.style.use('seaborn')

plt.figure(figsize=(12,8), tight_layout=True)
for l in [70]:
    pnl = np.load(f'/mnt/saved_data/PnL/pnl_AvellanedaLee({l}days)VOLOPT_TEST.npy')
    # pnl_vol = np.load(f'/mnt/saved_data/PnL/pnl_AvellanedaLee({l}days)VOLOPT_TESTx3.npy')
    plt.plot(pnl, 'red', linewidth=1.2, label=fr'$\tilde{{T}}$={l}, $\Lambda$=3%')
    # plt.plot(pnl, 'darkred', linewidth=1.2, label=fr'$\tilde{{T}}$={l}, $\Lambda$=2.5%', alpha=0.3)

plt.plot(np.load(f'/mnt/saved_data/PnL/pnl_spy_test.npy'), 'k', linewidth=1.2, label='First PC')
trading_days = np.array(pd.read_pickle('/mnt/saved_data/PriceData.pkl').Date)[4029+9:]
print(len(trading_days)-252)
print([trading_days[i] for i in np.arange(251, len(trading_days), 61)])
# print(len(trading_days))
# tickers = pd.read_pickle('/mnt/saved_data/returns/ReturnsData.pkl').columns.to_list()
# x_quantity = 61
# x_label_position = np.arange(252, len(trading_days)+191, x_quantity)
# print(x_label_position)
# x_label_day = [trading_days[i] for i in x_label_position]
# print(x_label_day)
plt.xticks(np.arange(0, len(trading_days)-251, 61), [trading_days[i] for i in np.arange(251, len(trading_days), 61)], fontsize=12, rotation=90)
plt.legend()


# colors = ['blue', 'green', 'red', 'purple', 'gold', 'deepskyblue']
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6), tight_layout=True)
# for len, color, i in zip([50,60,70,80,90,100], colors, [1,2,3,4,5,6]):
#     tau = 1/np.load(f'/home/danielemdn/Documents/saved_data/kappas/kappas{len}.npy').flatten()
#     ax = plt.subplot(2,3,i)
#     y, x, _ = ax.hist(tau, bins=10000, label=f'{len}', color=color, alpha=0.60)
#     ax.vlines(len/504, 0, y.max(), colors='k', linestyles='dashed', alpha=0.7)
#     ax.vlines(tau.mean(), 0, y.max(), colors=color, linestyles='solid', alpha=0.8)
#     plt.xticks(np.arange(-0.1, 0.3, 0.05), np.arange(-0.1*252, 0.3*252, 0.05*252, dtype=int))
#     ax.set_xlim(-0.1, 0.3)
#     ax.set_xlabel('Days')
#     ax.legend([fr'$\tilde{{T}}$ = {len}', f'Filter = {int(len/504 * 252)}', f'Mean = {int(tau.mean()*252)}'])
# plt.savefig(f'../dist_tau.png')

# plt.figure()
# for len in [50,60,70,80,90,100]:
#     pnl = np.load(f'/mnt/saved_data/PnL/pnl_AvellanedaLee({len}days)MOD.npy')
#     plt.plot(pnl, label=f'{len}MOD')
#
# plt.legend()



#xi = np.load('/mnt/saved_data/AR_res80.npy')
#std_xi = np.zeros(shape=(xi.shape[0], xi.shape[1]))

#for i in tqdm(range(xi.shape[0])):
#	for j in range(xi.shape[1]):
#		std_xi[i,j] = xi[i,j,:].std()

#std_xi = std_xi.flatten()
#plt.hist(std_xi, bins=100)
plt.show()
