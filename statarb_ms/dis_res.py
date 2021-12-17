import numpy as np
import pandas as pd
from makedir import go_up
from scipy.stats import normaltest
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# res = np.load(go_up(1) + '/saved_data/dis_res.npy')
# df = pd.read_csv(go_up(1) + '/saved_data/ScoreData.csv')
# tickers = df.columns.to_list()
# pvalues = np.zeros(shape=df.shape)
#
# for day in tqdm(range(res.shape[0])):
#     for stock in range(res.shape[1]):
#         # print(day,stock)
#         p = normaltest(res[day,stock])[1]
#         if p >= 0.05:
#             pvalues[day,stock] = 1
#         else:
#             pvalues[day,stock] = 0
#
# df_pvalues = pd.DataFrame(pvalues, columns=tickers)
# df_pvalues.to_csv(go_up(1) + '/saved_data/df_pvalues.csv', index=False)

df_pvalues = pd.read_csv(go_up(1) + '/saved_data/df_pvalues.csv')
perc = np.zeros(shape=df_pvalues.shape[1])

for stock in df_pvalues.columns:
    stock_idx = df_pvalues.columns.get_loc(stock)
    perc[stock_idx] = df_pvalues[stock].sum()/df_pvalues.shape[0] * 100


dic = {stock: val for stock,val in zip(df_pvalues.columns, perc)}
dic = dict(sorted(dic.items(), key=lambda item: item[1]))
perc = pd.DataFrame(dic, index=np.arange(1))
print(f'Top 10: {perc.columns.to_list()[:10]} \nBottom 10: {perc.columns.to_list()[-10:]}')
plt.figure(figsize=(10,8))
plt.bar(np.arange(0, perc.shape[1]), perc.iloc[0], color='darkorange')
plt.bar(np.arange(0, perc.shape[1]), height=100-perc.iloc[0], bottom=perc.iloc[0], color='darkred')
# plt.xticks(np.arange(0, perc.shape[1]), perc.columns, fontsize=13, rotation=60)
plt.xlabel('Companies', fontsize=13)
plt.ylabel('Percentage', fontsize=13)
plt.legend(['Accepted', 'Rejected'], fontsize=11)
plt.show()


percentage = np.zeros(shape=25)
c = 0
for i in range(0, df_pvalues.shape[0], 252):
    period = df_pvalues[i:i+252]
    perc = np.array(period).sum().sum()/(364*252) * 100
    percentage[c] = perc
    c += 1

perc = np.array(df_pvalues[-253:]).sum().sum()/(253*364) * 100
percentage[-1] = perc
years = ['1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020' ]
dict = {stock: val for stock,val in zip(years, percentage)}
perc = pd.DataFrame(dict, index=np.arange(1))
plt.figure(figsize=(10,8))
plt.bar(np.arange(0, perc.shape[1]), perc.iloc[0], color='darkorange')
plt.bar(np.arange(0, perc.shape[1]), height=100-perc.iloc[0], bottom=perc.iloc[0], color='darkred')
plt.xticks(np.arange(0, perc.shape[1]), years, fontsize=13, rotation=60)
plt.xlabel('Years', fontsize=13)
plt.ylabel('Percentage', fontsize=13)
plt.legend(['Accepted', 'Rejected'], fontsize=11)
plt.show()
