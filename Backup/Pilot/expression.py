import variable
import random
import numpy as np
import random

class Expression:
	TYPES = [None]

	def __init__(self, lhs, operator_type = None, rhs = None):
		self.operator_type = operator_type
		self.lhs = lhs
		self.rhs = rhs
		self.str_lhs = self.lhs.name if type(self.lhs) in (variable.Variable, variable.Array) else str(self.lhs)	
		self.str_rhs = self.rhs.name if type(self.rhs) in (variable.Variable, variable.Array) else str(self.rhs)
		assert(self.is_valid())

	def format(self):
		return [self.str_lhs]

	def is_valid(self):
		if (self.operator_type == None):
			return True
		else:
			return False

	def delete(self):
		if (self.parent != None):
			self.parent.children.remove(self)

	def randomize(parent):
		value_vars = []
		array_vars = []
		for var in parent.vars:
			if (type(var) is variable.Variable):
				value_vars.append(var)
			elif (type(var) is variable.Array):
				array_vars.append(var)
		possible_classes = [Expression]
		if (len(value_vars) > 0):
			possible_classes.append(ValueExpression)
		if (len(array_vars) > 0):
			possible_classes.append(ArrayExpression)
		selected_class = random.choice(possible_classes)

		if (selected_class is Expression):
			gen_lhs = random.choice(parent.vars)
			gen_rhs = None
		elif (selected_class is ValueExpression):
			gen_lhs = random.choice(value_vars)	
			gen_rhs = random.choice((random.choice(value_vars), np.random.poisson(0.5, 1).item()))
		elif (selected_class is ArrayExpression):
			gen_lhs = random.choice(array_vars)
			gen_rhs = random.choice((random.choice(value_vars), np.random.poisson(0.5, 1).item()))
		else:
			raise NotImplementedError

		gen_operator_type = random.choice(selected_class.TYPES)

		return selected_class(gen_lhs, gen_operator_type, gen_rhs)

class ValueExpression(Expression):
	TYPES = ['+', '-', '*', '/', '%', '**', '//', '==', '!=', '>', '<', '>=', '<=', 'and', 'or']

	def format(self):
		return [self.str_lhs + ' ' + self.operator_type + ' ' + self.str_rhs]

	def is_valid(self):
		if (type(self.lhs) in (bool, int, str, float, list, tuple, variable.Variable)):
			return True
		else:
			return False

class ArrayExpression(Expression):
	TYPES = [('[', ']')]

	def format(self):
		return [self.str_lhs + ' ' + self.operator_type[0] + self.str_rhs + self.operator_type[1]]

	def is_valid(self):
		if (type(self.lhs) == variable.Array):
			return True
		else:
			return False

