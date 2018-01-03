import numpy as np

class Assignment():
	def __init__(self, config, variable, is_increment, lhs, operator = '', rhs = ''):
		self.config 		= config
		self.variable 		= variable
		self.is_increment 	= is_increment
		self.a_operator 	= ('+=' if self.is_increment else '=')
		self.lhs 			= lhs
		self.str_lhs 		= str(lhs)
		self.operator 		= operator
		self.rhs 			= rhs

	def __str__(self):
		if (self.operator == ''):
			return '%s %s %s' % (self.variable, self.a_operator, self.str_lhs)
		elif (self.config.OPERATORS[self.operator][1] == 1):
			return '%s %s %s (%s)' % (self.variable, self.a_operator, self.operator, self.lhs)
		elif (self.config.OPERATORS[self.operator][1] == 2):
			return '%s %s %s (%s, %s)' % (self.variable, self.a_operator, self.operator, self.lhs, self.rhs)

class Program:
	def __init__(self, config):
		self.config = config
		self.consts = {
			'loop': ['n', 'i', 'a'],
			'global': ['n']
		}
		self.vars = {
			'loop': [],
			'global': []
		}
		self.assignments = {
			'preloop': [],
			'loop': [],
			'postloop': []
		}
		self.prediction = Assignment(self.config, 'y_hat', None, None)
		self.fitness = None

	def get_mse(self, X, y):
		rss = self.get_rss(X, y)
		return rss / y.size

	def get_rss(self, X, y):
		indeces = np.random.choice(y.size, int(y.size * self.config.BATCH_SIZE), False)
		exec(str(self), globals())
		rss = sum(np.power(self.predict(X[m]) - y[m], 2) for m in indeces)
		return rss

	def __str__(self):
		# lines = []
		# lines.append('import numpy as np')
		# lines.append('np.seterr(all = "ignore")')
		# lines.append('def predict(X, n):')
		# for elm in self.assignments['preloop']:
		# 	lines.append('\t' + str(elm))
		# lines.append('\tfor i in range(n):')
		# lines.append('\t\tx = X[i]')
		# for elm in self.assignments['loop']:
		# 	lines.append('\t\t' + str(elm))
		# for elm in self.assignments['postloop']:
		# 	lines.append('\t' + str(elm))
		# lines.append('\treturn y_hat')

		lines = [
			'import numpy as np',
			'np.seterr(all = "ignore")',
			'def predict(X, n):',
			*map(lambda x: '\t' + str(x), self.assignments['preloop']),
			'\tfor i in range(n):',
			'\t\tx = X[i]',
			'\t\ta = 0.95**i',
			*map(lambda x: '\t\t' + str(x), self.assignments['loop']),
			*map(lambda x: '\t' + str(x), self.assignments['postloop']),
			'\treturn y_hat'
		]
		return '\n'.join(lines)

	def predict(self, X):
		try:
			y_hat = predict(X, len(X))
		except Exception as e:
			return np.inf
		return y_hat

	def reduce(self):
		used_variables = ['y_hat']
		for i in range(len(self.assignments['postloop']) - 1, -1, -1):
			assignment = self.assignments['postloop'][i]
			if (assignment.variable in used_variables):
				if (type(assignment.lhs) == str and '[' not in assignment.lhs and assignment.lhs not in ('', 'n', 'i') and assignment.lhs not in used_variables):
						used_variables.append(assignment.lhs)
				if (type(assignment.rhs) == str and '[' not in assignment.rhs and assignment.rhs not in ('', 'n', 'i') and assignment.rhs not in used_variables):
						used_variables.append(assignment.rhs)
		for i in range(len(self.assignments['loop']) - 1, -1, -1):
			assignment = self.assignments['loop'][i]
			if (assignment.variable in used_variables):
				if (type(assignment.lhs) == str and '[' not in assignment.lhs and assignment.lhs not in ('', 'n', 'i') and assignment.lhs not in used_variables):
						used_variables.append(assignment.lhs)
				if (type(assignment.rhs) == str and '[' not in assignment.rhs and assignment.rhs not in ('', 'n', 'i') and assignment.rhs not in used_variables):
						used_variables.append(assignment.rhs)
		for i in range(len(self.assignments['preloop']) - 1, -1, -1):
			assignment = self.assignments['preloop'][i]
			if (assignment.variable in used_variables):
				if (type(assignment.lhs) == str and '[' not in assignment.lhs and assignment.lhs not in ('', 'n', 'i') and assignment.lhs not in used_variables):
						used_variables.append(assignment.lhs)
				if (type(assignment.rhs) == str and '[' not in assignment.rhs and assignment.rhs not in ('', 'n', 'i') and assignment.rhs not in used_variables):
						used_variables.append(assignment.rhs)
		for section in self.assignments:
			replacement = []
			for assignment in self.assignments[section]:
				if (assignment.variable in used_variables):
					replacement.append(assignment)
			self.assignments[section] = replacement