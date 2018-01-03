class Variable:
	counter = 1
	def __init__(self, name = None):
		if (name == None):
			self.name = 'var_' + str(Variable.counter)
			Variable.counter += 1
		else:
			self.name = name

class Array:
	counter = 1
	def __init__(self, name = None):
		if (name == None):
			self.name = 'arr_' + str(Array.counter)
			Array.counter += 1
		else:
			self.name = name