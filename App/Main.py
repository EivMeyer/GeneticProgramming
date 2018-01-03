import pandas as pd
import numpy as np
import pprint as pprint
import os
import argparse
import shutil
import random
import sklearn.preprocessing

import Program
import Config
import Evolution
import Tree
import Datasets

# temporary function for testing predictor functions
def test_predict(X,n):
	y_hat = sum(np.divide(np.add(X[i][0], X[i][0]), np.exp(np.subtract(n, i))) for i in range(n))
	return y_hat
#

def reset_storage(root):
	report_path = os.path.normpath(root + '/report')
	if (os.path.exists(report_path)):
		print('Deleting report folder...')
		shutil.rmtree(report_path, ignore_errors=True)
	os.makedirs(report_path)
	os.makedirs(os.path.normpath(report_path + '/islands'))

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description='Genetic programming engine')
	parser.add_argument('-test', dest = 'should_test', help = 'Predict single test sample using defined predictor function', action = 'store_true')
	args = parser.parse_args()

	root = os.path.dirname(os.path.realpath(__file__))
	config = Config.Configuration()
	reset_storage(root)

	#X_train, y_train, X_val, y_val = Datasets.load(name = 'Random_Walk_With_Mommentum', config = config, root = root)
	X_train, y_train, X_val, y_val = Datasets.load(name = 'Bitcoin_USD', config = config, root = root)

	if (args.should_test):
		print(' & '.join(['{:<.2f}'.format(x[0]) for x in X_train[0].tolist()]))
		print(X_train[0], y_train[0], test_predict(X_train[0], len(X_train[0])))
		exit(0)

	evolution = Evolution.Evolution(root = root, config = config, X_train = X_train, y_train = y_train, X_val = X_val, y_val = y_val)