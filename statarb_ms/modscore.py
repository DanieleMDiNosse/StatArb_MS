import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

ret = pd.read_pickle('/mnt/saved_data/returns/ReturnsData.pkl')
for len in tqdm([50,60,70,90,100]):
	score = pd.read_pickle(f'/mnt/saved_data/scores/ScoreData{len}.pkl')
	kappa = np.load(f'/mnt/saved_data/kappas/kappas{len}.npy')
	alpha = np.load(f'/mnt/saved_data/alphas/alphas{len}.npy')
	sgm_eq = np.load(f'/mnt/saved_data/sgm_eqs/sgm_eq{len}.npy')
	score_mod = np.zeros_like(sgm_eq)

	score_values = score.values

	for day in range(score_mod.shape[0]):
		for col in range(score_mod.shape[1]):
			if kappa[day, col]*sgm_eq[day,col] == 0:
				score_mod[day, col] = score_values[day, col]
	#			print(kappa[day,col], sgm_eq[day,col])
			else:
				score_mod[day, col]  = score_values[day,col] - (alpha[day,col]/252)/(kappa[day,col]*sgm_eq[day,col])

	score_mod = pd.DataFrame(score_mod, index=range(score_mod.shape[0]), columns=score.columns)
	score_mod.to_pickle(f'/mnt/saved_data/scores/ScoreModData{len}.pkl')
