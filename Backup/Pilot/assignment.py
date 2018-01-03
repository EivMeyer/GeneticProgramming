import variable
import expression
import random

class Assignment:
	def __init__(self, var, expression, index = None):
		self.var = var
		self.expression = expression
		self.index = index
		self.str_index = self.index.name if type(self.index) is (variable.Variable) else str(self.index)
		self.parent = None	
		assert(self.is_valid())

	def format(self):
		if (type(self.var) is variable.Array and self.index != None):
			return [self.var.name + '[' + self.str_index + '] = ' + self.expression.format()[0]]
		else:
			return [self.var.name + ' = ' + self.expression.format()[0]]

	def is_valid(self):
		if (type(self.var) is not variable.Array and self.index != None):
			return False
		return True

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

		gen_var = random.choice(parent.vars)
		gen_expression = expression.Expression.randomize(parent)

		if (type(gen_var) is variable.Variable):
			gen_index = None
		elif (type(gen_var) is variable.Array):
			gen_index = random.choice((random.choice(value_vars), 0, -1))
		else:
			raise NotImplementedError

		return Assignment(gen_var, gen_expression, gen_index)
		