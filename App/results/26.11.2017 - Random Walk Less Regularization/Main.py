import pandas as pd
import numpy as np
import pprint as pprint
import os
import argparse
import shutil
import random
import sklearn.preprocessing

# -----------------

import Program
import Config
import Evolution
import Tree

def reset_storage(root):
	report_path = os.path.normpath(root + '/report')
	if (os.path.exists(report_path)):
		print('Deleting report folder...')
		shutil.rmtree(report_path, ignore_errors=True)
	os.makedirs(report_path)
	os.makedirs(os.path.normpath(report_path + '/islands'))

def predict(X, n):
	const_209370883351446112497346724238 = -0.0026758271435904893
	const_209370883351446112497346724238 = -0.002473209240739025
	const_209370883351446112497346724238 = -0.00263973526207381
	const_209370883351446112497346724238 = -0.0031364290862955055
	const_209370883351446112497346724238 = -0.0037380046409053224
	y_hat = sum(np.sin(np.multiply(np.subtract(np.cos(X[i][0]), n), np.multiply(X[i][0], const_209370883351446112497346724238))) for i in range(n))
	return y_hat

def generate_sample():

	# Randomly Normally distributed
	# mu = np.random.normal(0, 1)
	# sigma = np.random.gamma(abs(mu)/10, 10)
	# X = []
	# y = 0
	# for i in range(10):
	# 	x = []
	# 	for k in range(config.K):
	# 		rn = np.random.normal(mu, sigma)
	# 		x.append(rn)
	# 	x = np.array(x)
	# 	X.append(x)
	# X = np.array(X)
	# y = X.mean()
	# return X, y

	# Autocorrelated
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
	y_pred = predict(X, len(X))
	return X, y

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description='Genetic programming engine')
	
	args = parser.parse_args()

	root = os.path.dirname(os.path.realpath(__file__))
	config = Config.Configuration()
	reset_storage(root)

	X_train = []
	y_train = []
	X_val 	= []
	y_val 	= []
	for m in range(1000):
		X_m, y_m = generate_sample()
		X_train.append(X_m)
		y_train.append(y_m)
	for m in range(10000):
		X_m, y_m = generate_sample()
		X_val.append(X_m)
		y_val.append(y_m)
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_val = np.array(X_val)
	y_val = np.array(y_val)

	evolution = Evolution.Evolution(root = root, config = config, X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val)