import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# sp500cost = pd.read_pickle('../saved_data/SP500histcost_matched.pkl')
# trading_days = pd.read_pickle('../saved_data/ReturnsDataHuge.pkl').shape[0] - 252
ret = pd.read_pickle('/mnt/saved_data/ReturnsDataHuge.pkl')
ret_old = pd.read_pickle('/mnt/saved_data/ReturnsData.pkl')

l = []
for col in ret.columns:
    for val in ret[col]:
        if val > 2.0:
            print(col, ret[ret[col] == val].index.values)
            l.append(col)
        if val < -2.0:
            l.append(col)
l = list(set(l))

ret = ret.drop(columns=l)
ret = ret.fillna(value=0.0)
print(ret.shape)
# ret.to_pickle('/mnt/saved_data/ReturnsDataHugeCleanedWM.pkl')
