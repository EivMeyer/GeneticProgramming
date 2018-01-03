import numpy as np
import Tree

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
	Counter = 0
	def __init__(self, type, config):
		self.type = type
		self.config = config
		self.consts = {
			'loop': ['n', 'i'],
			'global': ['n']
		}
		self.vars = {
			'loop': [],
			'global': []
		}
		self.assignments = {
			'preloop': [],
			'postloop': []
		}
		self.loop_tree = None
		self.fitness = None
		self.id = 'program_' + str(Program.Counter)
		Program.Counter += 1

	def get_mse(self, X, y, use_batch):
		rss = self.get_rss(X, y, use_batch)
		return rss / y.size

	def get_rss(self, X, y, use_batch):
		if (use_batch):
			indeces = np.random.choice(y.size, int(y.size * self.config.BATCH_SIZE), False)
		else:
			indeces = [m for m in range(y.size)]
		exec(str(self), globals())
		rss = sum(np.power(self.predict(X[m]) - y[m], 2) for m in indeces)
		return rss

	def rec_reduce(self, node = None, first_caller = True):
		used_consts = []
		if (node == None):
			node = self.loop_tree
		if (type(node.left_child) is Tree.Node):
			used_consts.extend(self.rec_reduce(node.left_child, False))
		elif ('const' in node.left_child):
			used_consts.append(node.left_child)
		if (type(node.right_child) is Tree.Node):
			used_consts.extend(self.rec_reduce(node.right_child, False))
		elif ('const' in node.right_child):
			used_consts.append(node.right_child)
		if (first_caller):
			self.consts = {
				'loop': ['n', 'i'],
				'global': ['n']
			}
			assignment_list_replacement = []
			for assignment in self.assignments['preloop']:
				if (assignment.variable in used_consts and assignment.variable not in assignment_list_replacement):
					assignment_list_replacement.append(assignment)
					self.consts['loop'].append(assignment.variable)
					self.consts['global'].append(assignment.variable)
			self.assignments['preloop'] = assignment_list_replacement
		else:
			return used_consts

	def __str__(self):
		lines = [
			#'import numpy as np',
			'np.seterr(all = "ignore")',
			'def predict(X, n):',
			# '\ty_hat = 0',
			*map(lambda x: '\t' + str(x), self.assignments['preloop']),
			# '\tfor i in range(n):',
			# '\t\tx = X[i]',
			# '\t\ty_hat += ' + str(self.loop_tree),
			'\ty_hat = sum(' + str(self.loop_tree) + ' for i in range(n))',
			'\treturn y_hat'
		]
		return '\n'.join(lines)

	def predict(self, X):
		try:
			y_hat = predict(X, len(X))
		except Exception as e:
			#raise e
			return np.inf
		return y_hat
