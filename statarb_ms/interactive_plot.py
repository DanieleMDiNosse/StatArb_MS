import plotly.express as px
import pandas as pd
import numpy as np

df = pd.read_csv('../saved_data/PriceData.csv')
pnl_gas = np.load('../saved_data/pnl_gas.npy')
spy_ret = np.load('../saved_data/spy_ret.npy')
pnl = np.load('../saved_data/pnl.npy')
PnL = pd.DataFrame({'Date': df.Date[253:], 'pnl_gas': pnl_gas, 'pnl': pnl, 'spy_ret': spy_ret})
fig = px.line(PnL, x="Date", y=["pnl", "pnl_gas", "spy_ret"], title='PnL')
fig.write_html('pnl.html')



pnlfirstyears = np.load('../saved_data/pnl_firstyears.npy')
PnL = pd.DataFrame({'Date': df.Date[252:1259+1], 'pnl2007': pnlfirstyears, 'pnl':pnl[:pnlfirstyears.shape[0]]+0.01})
fig = px.line(PnL, x='Date', y=['pnl2007', 'pnl'])
fig.write_html('pnl2007.html')
