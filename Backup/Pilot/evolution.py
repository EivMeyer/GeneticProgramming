import program
import numpy as np
import block
import variable
import statement
import assignment
import root
import expression
import randomizer
import math

class Evolution:
	def __init__(self, user_options = {}):
		self.options = Evolution.default_options()
		for key in user_options:
			if (key not in self.options):
				raise Exception('Unknown option specified: ' + key)
			self.options[key] = user_options[key]

		self.highscore = []

		def input_generator():
			return 4

		def score(z):
			return 2 / (1 + math.exp(0.01*math.pow(20-z, 2)))

		for k in range(self.options['population_count']):
			variable_count = self.options['variable_count']
			initial_program_size = int(np.random.normal(self.options['initial_program_mean_size'], self.options['initial_program_size_std'], 1).item())

			p = program.Program(input_generator, score)

			for i in range(variable_count):
				p.root.insert(assignment.Assignment(variable.Variable(), expression.Expression(0)))

			for i in range(initial_program_size):
				selected_parent = randomizer.get_random_parent(p.root)
				gen_element = randomizer.create_random_element(selected_parent)
				selected_parent.insert(gen_element)

			if (not p.has_infinite_loop()):
				print(k, p.run())
				try:
					pass
				except Exception as e:
					continue
					print(k, e)

	def default_options():
		return {
			'population_count': 				100,
			'initial_program_mean_size': 		10,
			'variable_count': 					1,
			'initial_program_size_std': 		2,
		}
