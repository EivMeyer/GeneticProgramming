import pandas as pd
import numpy as np
import pprint as pprint
import random

# -----------------

import Program
import Config
import Evolution

def generate_sample():

	# Randomly Normally distributed
	# mu = np.random.normal(0, 10)
	# sigma = np.random.gamma(abs(mu)/10, 10)
	# X_m = []
	# for i in range(50):
	# 	x_i = []
	# 	for k in range(config.K):
	# 		x_i.append(np.random.normal(mu, sigma))
	# 	x_i = np.array(x_i)
	# 	X_m.append(x_i)
	# X_m = np.array(X_m)
	# return X_m

	# Autocorrelated
	x = np.random.normal(0, 10)
	v = np.random.normal(0, 1)
	X = []
	for i in range(50):
		X.append(x)
		x += v
		v += np.random.normal(0, 0.5)
	X = np.array(X)
	y = x

	return X, y

def get_y_m(X_m):
	num = 0
	den = 0
	for i in range(X_m.size):
		alpha = 0.95**i
		num += X_m[i].item() * alpha
		den += alpha
	return num/den

# s = pd.read_csv('./data/hohenpeissenb_temp.csv')
# print(s[['YEAR', 'JUL']])

# exit(0)

if (__name__ == '__main__'):
	config = Config.Configuration()

	X_train = []
	y_train = []
	for m in range(100):
		X_m, y_m = generate_sample()
		X_train.append(X_m)
		y_train.append(y_m)
	X_train = np.array(X_train)
	y_train = np.array(y_train)

	evolution = Evolution.Evolution(config = config, X_train = X_train, y_train = y_train)