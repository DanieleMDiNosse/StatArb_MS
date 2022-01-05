import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from makedir import go_up

trading_days = pd.read_csv(go_up(1) + '/saved_data/PriceData.csv').Date
x_label_position = np.arange(0, len(trading_days)-252, 10)
x_label_day = [trading_days[i] for i in x_label_position]

ret = np.load(go_up(1) + '/saved_data/prova.npy')
plt.plot(ret)
plt.xticks(x_label_position, x_label_day, rotation=60)
plt.show()
