import random

class Statement:
	TYPES = ['pass']
	def __init__(self, statement_type):
		assert (statement_type in Statement.TYPES)
		self.statement_type = statement_type
		self.parent = None

	def format(self):
		return (self.statement_type,)

	def delete(self):
		if (self.parent != None):
			self.parent.children.remove(self)

	def randomize(parent):
		statement_type = random.choice(Statement.TYPES)
		return Statement(statement_type)