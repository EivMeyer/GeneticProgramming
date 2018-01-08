import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

def load(name, config, root):
	assert (name in {'Random_Walk_With_Mommentum', 'North_Carolina_Weather', 'Bitcoin_USD'})
	
	if (name == 'Random_Walk_With_Mommentum'):
		X_train = []
		y_train = []
		X_val 	= []
		y_val 	= []
		for m in range(1000):
			X_m, y_m = generate_random_walk_sample(config)
			X_train.append(X_m)
			y_train.append(y_m)
		for m in range(10000):
			X_m, y_m = generate_random_walk_sample(config)
			X_val.append(X_m)
			y_val.append(y_m)
		X_train = np.array(X_train)
		y_train = np.array(y_train)
		X_val = np.array(X_val)
		y_val = np.array(y_val)

		return X_train, y_train, X_val, y_val

	elif (name in 'North_Carolina_Weather'):
		df = pd.read_csv(os.path.normpath(root + '/data/rdu-weather-history.csv'), header = 0, delimiter = ';')
		all_days = []
		for row in df.iterrows():
			day = row[1]
			if (not math.isnan(day['temperaturemax'])):
				all_days.append({
					'date': datetime.strptime(day['date'], '%Y-%m-%d'),
					'max_temp': day['temperaturemax'],
					'min_temp': day['temperaturemin'],
					'avg_temp': (day['temperaturemax'] + day['temperaturemin']) / 2
				})
		all_days = sorted(all_days, key = lambda x: x['date'])
		X = []
		y = []
		last_days = []
		for day in all_days:
			if (len(last_days) > 40):
				last_days.pop(0)
				x_i = []
				for hist_day in last_days:
					x_i.append(np.asarray([hist_day['avg_temp']]))
				x_i = np.asarray(x_i)
				X.append(x_i)
				y.append(day['avg_temp'])
			last_days.append(day)
		X = np.asarray(X)
		y = np.asarray(y)

		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

		return X_train, y_train, X_val, y_val

	elif (name in 'Bitcoin_USD'):
		df = pd.read_csv(os.path.normpath(root + '/data/btc-usd.csv'), header = 0, delimiter = ',')

		all_days = []
		for row in df.iterrows():
			day = row[1]
			if (not math.isnan(day['Close'])):
				all_days.append({
					'date': datetime.strptime(day['Date'], '%Y-%m-%d'),
					'close': day['Close']
				})
		all_days = sorted(all_days, key = lambda x: x['date'])
		X = []
		y = []
		last_days = []
		for day in all_days:
			if (len(last_days) > 100):
				last_days.pop(0)
				x_i = []
				for hist_day in last_days:
					x_i.append(np.asarray([hist_day['close']]))
				x_i = np.asarray(x_i)
				X.append(x_i)
				y.append(day['close'])
			last_days.append(day)
		X = np.asarray(X)
		y = np.asarray(y)

		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

		return X_train, y_train, X_val, y_val		

	else:
		raise Exception('Undefined dataset requested: ' + name)

def generate_random_walk_sample(config):
	p = np.random.normal(0, 1)
	v = np.random.normal(0, 0.1)
	X = []
	for i in range(20):
		x = []
		p += v
		for k in range(config.K):
			x.append(np.random.normal(p, abs(p/10)))
		v += np.random.normal(0, 0.02)
		x = np.array(x)
		X.append(x)
	X = np.array(X)
	y = np.random.normal(p, abs(p/10))

	return X, y
