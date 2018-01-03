import statement
import variable
import assignment
import expression
import block
import random

class Root:
	def __init__(self):
		self.children = [statement.Statement('pass')]
		self.vars = []
		self.parent = None

	def format(self):
		lines = ['def main(x):', '\ty = 0']
		for child in self.children:
			child_lines = child.format()
			for child_line in child_lines:
				lines.append('\t' + child_line)
		lines.append('\treturn int(y)')
		return lines

	def insert(self, element):
		if (type(element) is assignment.Assignment):
			if (element.var not in self.vars):
				self.vars.append(element.var)

		element.parent = self
		self.children.append(element)

		self.propagate()

	def propagate(self):
		for child in self.children:
			if (type(child) is block.Block):
				for var in self.vars:
					if (var not in child.vars):
						child.vars.append(var)
				child.propagate()

	def has_infinite_loop(self):
		ans = False
		for child in self.children:
			if (type(child) is block.Block):
				if (child.has_infinite_loop()):
					return True
		return ans
