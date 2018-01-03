import random
import assignment
import block
import expression
import statement
import variable

def get_random_element(root):
	parent = get_random_parent(root)
	if (len(parent.children) > 0):
		return random.choice(parent.children)
	else:
		return parent

def get_random_parent(root):
	cur = root
	while (random.random() > 0.5):
		block_children = []
		for child in cur.children:
			if (type(child) is block.Block):
				block_children.append(child)
		if (len(block_children) == 0):
			break
		else:
			cur = random.choice(block_children)
	return cur

def create_random_element(parent):
	classes = (assignment.Assignment, block.Block) #, statement.Statement)
	selected_class = random.choice(classes)
	gen_element = selected_class.randomize(parent)
	return gen_element