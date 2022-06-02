import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('seaborn')

# plt.figure(figsize=(12,8), tight_layout=True)
# for len in [50,60,70,80]:
#     pnl = np.load(f'/home/danielemdn/Documents/saved_data/PnL/pnl_AvellanedaLee({len}days).npy')
#     pnl_mod = np.load(f'/home/danielemdn/Documents/saved_data/PnL/pnl_AvellanedaLee({len}days)OPT.npy')
#     plt.plot((pnl_mod-pnl)/pnl, linewidth=1.2, label=f'{len}')
# plt.legend()


colors = ['blue', 'green', 'red', 'purple', 'gold', 'deepskyblue']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6), tight_layout=True)
for len, color, i in zip([50,60,70,80,90,100], colors, [0,1,2,3,4,5]):
    tau = 1/np.load(f'/home/danielemdn/Documents/saved_data/kappas/kappas{len}.npy').flatten()
    y, x, _ = axs[i].hist(tau, bins=5000, label=f'{len}', color=color, alpha=0.60)
    axs[i].vlines(len/504, 0, y.max(), colors='k', linestyles='dashed', alpha=0.7)
    axs[i].xticks(np.arange(-0.1, 0.3, 0.05), np.arange(-0.1*252, 0.3*252, 0.05*252, dtype=int))
    axs[i].xlim(-0.1, 0.3)
    axs[i].xlabel('Days')
    axs[i].legend()
    axs[i].savefig(f'../dist_tau{len}.png')

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
