import statement
import variable
import assignment
import expression
import random

class Block:
	TYPES = ['while', 'if']
	def __init__(self, block_type, condition):
		assert (block_type in Block.TYPES)
		self.block_type = block_type
		self.condition = condition
		self.children = [statement.Statement('pass')]
		self.vars = []
		self.parent = None

	def format(self):
		lines = [self.block_type + '(' + self.condition.format()[0] + '):']
		for child in self.children:
			child_lines = child.format()
			for child_line in child_lines:
				lines.append('\t' + child_line)
		return lines

	def insert(self, element):
		if (type(element) is assignment.Assignment):
			if (element.var not in self.vars):
				self.vars.append(element.var)

		element.parent = self
		self.children.append(element)

		self.propagate()

	def delete(self):
		if (self.parent != None):
			self.parent.children.remove(self)

	def propagate(self):
		for child in self.children:
			if (type(child) is Block):
				for var in self.vars:
					if (var not in child.vars):
						child.vars.append(var)
				child.propagate()

	def has_infinite_loop(self):
		if (self.block_type == 'while'):
			ans = True
			for child in self.children:
				if (type(child) is assignment.Assignment):
					if (child.var in (self.condition.lhs, self.condition.rhs)):
						ans = False
						break
				elif (type(child) is Block):
					if (child.has_infinite_loop):
						return True
				# else:
				# 	raise NotImplementedError
			return ans
		else:
			ans = False
			for child in self.children:
				if (type(child) is Block):
					if (child.has_infinite_loop()):
						return True
			return ans

	def randomize(parent):
		gen_block_type = random.choice(Block.TYPES)
		gen_condition = expression.Expression.randomize(parent)
		return Block(gen_block_type, gen_condition)
		print(45)

