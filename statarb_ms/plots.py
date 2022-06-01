import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('seaborn')

plt.figure(figsize=(12,8), tight_layout=True)
for len in [50,60,70,80,90,100]:
    pnl = np.load(f'/home/danielemdn/Documents/saved_data/PnL/pnl_AvellanedaLee({len}days).npy')
    pnl_mod = np.load(f'/home/danielemdn/Documents/saved_data/PnL/pnl_AvellanedaLee({len}days)MOD.npy')
    plt.plot((pnl_mod-pnl)/pnl, linewidth=1.2, label=f'{len}')
plt.legend()

# plt.figure()
# for len in [50,60,70,80,90,100]:
#     pnl = np.load(f'/mnt/saved_data/PnL/pnl_AvellanedaLee({len}days)MOD.npy')
#     plt.plot(pnl, label=f'{len}MOD')
#
# plt.legend()

plt.show()

#xi = np.load('/mnt/saved_data/AR_res80.npy')
#std_xi = np.zeros(shape=(xi.shape[0], xi.shape[1]))

#for i in tqdm(range(xi.shape[0])):
#	for j in range(xi.shape[1]):
#		std_xi[i,j] = xi[i,j,:].std()

#std_xi = std_xi.flatten()
#plt.hist(std_xi, bins=100)
#plt.show()
