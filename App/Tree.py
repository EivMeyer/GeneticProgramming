class Node:
	def __init__(self, operator, left_child, right_child, transformation = None):
		self.operator 		= operator
		self.left_child 	= left_child
		self.right_child 	= right_child
		self.transformation = transformation

	def pprint(self, level = 0):
		l = self.left_child.pprint(level + 1) if type(self.left_child) is Node else str(self.left_child)
		r = self.right_child.pprint(level + 1) if type(self.right_child) is Node else str(self.right_child)
		if (self.transformation == None):
			return '\n' + level*'\t' + l + '\n' + level*'\t' + self.operator + '\n' + level*'\t' + r
		else:
			return '\n' + level*'\t' + transformation + '(\n' + level*'\t' + l + '\n' + level*'\t' + self.operator + '\n' + level*'\t' + r + ')'

	def __str__(self):
		if (self.transformation == None):
			return '%s(%s, %s)' % (self.operator, str(self.left_child), str(self.right_child))
		else:
			return '%s(%s(%s, %s))' % (self.transformation, self.operator, str(self.left_child), str(self.right_child))

	def rec_get_node_count(self, counter = 0):
		if (type(self.left_child) is Node):
			counter += self.left_child.rec_get_node_count(counter)
		else:
			counter += 1
		if (type(self.right_child) is Node):
			counter += self.right_child.rec_get_node_count(counter)
		else:
			counter += 1
		return counter

	def rec_get_height(self, level = 0):
		if (type(self.left_child) is Node):
			left_height = self.left_child.rec_get_height(level)
		else:
			left_height = 0
		if (type(self.right_child) is Node):
			right_height = self.right_child.rec_get_height(level)
		else:
			right_height = 0
		return max(left_height, right_height) + 1


	

