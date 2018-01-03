import block
import variable
import statement
import assignment
import root
import expression
import randomizer

class Program:
	def __init__(self, input_generator, score):
		self.input_generator = input_generator
		self.score = score
		self.root = root.Root()
		self.x = variable.Variable('x')
		self.y = variable.Variable('y')
		self.root.vars.append(self.x)
		self.root.vars.append(self.y)

	def save(self):
		with open('script.py', 'w') as file:
			file.write(self.format())

	def format(self):
		return '\n'.join(self.root.format())

	def run(self):
		print(self.format())
		exec(self.format(), globals())
		try:
			y = main(self.input_generator())
		except Exception as e:
			#print(e)
			return 0
		return  self.score(y)

	def has_infinite_loop(self):
		return self.root.has_infinite_loop()