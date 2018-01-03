import Program
import numpy as np
import random
import sys
import time

class Evolution():
	def __init__(self, config, X_train, y_train):
		self.config 	= config
		self.X_train 	= X_train
		self.y_train 	= y_train
		self.tss 		= sum((y_train - y_train.mean())**2)
		self.var_y 		= y_train.var()
		self.population = []
		self.pop_dict 	= {}
		self.optimize()

	def get_evolved_program(self, parent):
		child = Program.Program(self.config)
		has_changed = False

		for assignment in parent.assignments['preloop']:
			if (random.random() > self.config.P_DELETE_LINE):
				if ('const' in assignment.variable):
					const = assignment.variable
					if (random.random() < self.config.P_MODIFY_LINE):
						has_changed = True
						value = self.create_random_const()
					else:
						value = assignment.lhs
					child_assignment = Program.Assignment(self.config, const, False, value)
					child.consts['loop'].append(const)
					child.consts['global'].append(const)
					child.assignments['preloop'].append(child_assignment)
					while (random.random() < self.config.P_ADD_LINE):
						has_changed = True
						value = self.create_random_const()
						const = 'const_' + str(len(child.consts['global']) - 1)
						child.consts['loop'].append(const)
						child.consts['global'].append(const)
						added_assignment = Program.Assignment(self.config, const, False, value)
						child.assignments['preloop'].append(added_assignment)
				else:
					var = assignment.variable
					if (random.random() < self.config.P_MODIFY_LINE):
						has_changed = True
						init_var_as_const = np.random.random() < self.config.P_INIT_VAR_AS_CONST
						if (init_var_as_const):
							child_assignment = self.create_random_assignment('preloop', [], child.consts['global'], False, var)
						else:
							child_assignment = assignment
					else:
						child_assignment = assignment
					child.vars['global'].append(var)
					child.assignments['preloop'].append(child_assignment)
					while (random.random() < self.config.P_ADD_LINE):
						has_changed = True
						var = 'gvar_' + str(len(child.vars['global']))
						init_var_as_const = np.random.random() < self.config.P_INIT_VAR_AS_CONST
						if (init_var_as_const):
							added_assignment = self.create_random_assignment('preloop', [], child.consts['global'], False, var)
						else:
							added_assignment = Program.Assignment(self.config, var, False, 0)
						child.vars['global'].append(var)
						child.assignments['preloop'].append(added_assignment)
		if (len(child.vars['global']) == 0):
			return (parent, False)
		for assignment in parent.assignments['loop']:
			var = assignment.variable
			if ('var' in var):
				if ('lvar' in var and random.random() > self.config.P_DELETE_LINE):
					if (random.random() < self.config.P_MODIFY_LINE):
						has_changed = True
						child_assignment = self.create_random_assignment('loop', child.vars['loop'], child.consts['loop'], False, var)
					else:
						child_assignment = assignment
					child.vars['loop'].append(var)
					child.assignments['loop'].append(child_assignment)
					while (random.random() < self.config.P_ADD_LINE):
						has_changed = True
						var = 'lvar_' + str(len(child.vars['loop']))
						child_assignment = self.create_random_assignment('loop', child.vars['loop'], child.consts['loop'], False, var)
						child.vars['loop'].append(var)
						child.assignments['loop'].append(child_assignment)
				elif ('gvar' in var and var in child.vars['global']):
					if (random.random() < self.config.P_MODIFY_LINE):
						has_changed = True
						child_assignment = self.create_random_assignment('loop', child.vars['loop'], child.consts['loop'], True, var)
					else:
						child_assignment = assignment
					child.assignments['loop'].append(assignment)
			else:
				child.assignments['loop'].append(assignment)
		
		for assignment in parent.assignments['postloop']:
			var = assignment.variable
			if ('gvar' in var and random.random() > self.config.P_DELETE_LINE):
				if (random.random() < self.config.P_MODIFY_LINE):
					has_changed = True
					child_assignment = self.create_random_assignment('postloop', child.vars['global'], child.consts['global'], False, var)
				else:
					child_assignment = assignment
				child.assignments['postloop'].append(child_assignment)
				while (random.random() < self.config.P_ADD_LINE):
					has_changed = True
					var = 'gvar_' + str(np.random.randint(0, len(child.vars['global'])))
					added_assignment = self.create_random_assignment('postloop', child.vars['global'], child.consts['global'], False, var)
					child.assignments['postloop'].append(added_assignment)
			else:
				child.assignments['postloop'].append(assignment)

		if (not has_changed):
			return (parent, False)

		child.reduce()
		child_mse = child.get_mse(self.X_train, self.y_train)
		child.fitness = self.get_fitness(child_mse, child)
		child.mse = child_mse

		if (child.fitness > parent.fitness):
			return (child, True)
		else:
			return (parent, False)

	def mate(self, mother, father):
		assert(len(father.vars['global']) == len(mother.vars['global']))
		child = Program.Program(self.config)
		for assignment in father.assignments['preloop']:
			if ('const' in assignment.variable and type(assignment.lhs) is float):
				replacement = Program.Assignment(self.config, assignment.variable, False, np.random.normal(assignment.lhs, abs(assignment.lhs)/10))
				child.assignments['preloop'].append(replacement)
			else:
				child.assignments['preloop'].append(assignment)
		for assignment in mother.assignments['loop']:
			child.assignments['loop'].append(assignment)
		for assignment in father.assignments['postloop']:
			child.assignments['postloop'].append(assignment)

		child.reduce()
		mse = child.get_mse(self.X_train, self.y_train)
		child.fitness = self.get_fitness(mse, child)
		child.mse = mse

		return child

	def create_random_const(self):
		is_integer = np.random.random() < self.config.P_CONST_IS_INTEGER
		const = np.random.normal(self.config.CONST_EV, self.config.CONST_SD)
		if (is_integer):
			const = int(const)
		return const

	def create_random_assignment(self, context, vars, consts, is_increment, var = None):
		assert(context in ('preloop', 'loop', 'postloop'))
		assert(len(vars) > 0 or len(consts) > 0)
		assert(not(var == None and len(vars) == 0))

		if (var == None):
			var = random.choice(vars)

		is_possible = {
			'var': len(vars) > 0,
			'const': len(consts) > 0,
			'x': True
		}
		if (self.config.P_CHOOSE_CONST >= np.random.random() and is_possible['const'] or (not is_possible['var'] and context != 'loop')):
			lhs_type = 'const'
		else:
			if (is_possible['var']):
				if (self.config.P_CHOOSE_X >= np.random.random() and context == 'loop'):
					lhs_type = 'x'
				else:
					lhs_type = 'var'
			else:
				assert(context == 'loop')
				lhs_type = 'x'
		assert(lhs_type in ('var', 'const', 'x'))
		lhs = random.choice(consts) if lhs_type == 'const' else ('x[' + str(np.random.randint(0, self.config.K)) + ']' if lhs_type == 'x' else random.choice(vars))

		r = np.random.random()
		upto = 0
		for operator in self.config.OPERATOR_PROBABILITIES:
			if (upto + self.config.OPERATOR_PROBABILITIES[operator] >= r):
				break
			upto += self.config.OPERATOR_PROBABILITIES[operator]
		else:
			assert False, 'Should not get here'

		if (operator == ''):
			assignment = Program.Assignment(self.config, var, is_increment, lhs)
		else:
			if (len(vars) == 0):
				is_rhs_const = True
			elif (len(consts) == 0):
				is_rhs_const = False
			else:
				is_rhs_const = np.random.random() < self.config.P_CHOOSE_CONST
			rhs = random.choice(consts) if is_rhs_const else random.choice(vars)
			assignment = Program.Assignment(self.config, var, is_increment, lhs, operator, rhs)

		return assignment

	def get_fitness(self, mse, program):
		r_squared	   = 1 - mse / self.var_y
		program_length = len(program.assignments['preloop']) + len(program.assignments['loop']) + len(program.assignments['postloop'])
		variable_count = len(program.consts['loop']) + len(program.vars['global']) + len(program.vars['loop'])
		correctness    = 1 if mse == 0 else 0
		z = self.config.FITNESS_WEIGHTS['r_squared'] 		* r_squared 		+ \
		    self.config.FITNESS_WEIGHTS['program_length'] 	* program_length 	+ \
		    self.config.FITNESS_WEIGHTS['variable_count'] 	* variable_count    + \
		    self.config.FITNESS_WEIGHTS['correctness'] 		* correctness
		return 1/(1 + np.exp(-z))

	def optimize(self):
		for i in range(self.config.POPULATION_SIZE):
			sys.stdout.write('Creating program ' + str(i + 1) + ' / ' + str(self.config.POPULATION_SIZE) + '...\r')
			sys.stdout.flush()
			program = self.create_random_program()
			if (np.isfinite(program.fitness)):
				successive_standstills = 0
				while (successive_standstills <= self.config.MAX_SELF_EVOLUTION_ITER):
					program, is_updated = self.get_evolved_program(program)
					if (is_updated):
						successive_standstills = 0
					else:
						successive_standstills += 1
				self.population.append(program)
		sys.stdout.write('\t\t\t\t\t\t\r')
		sys.stdout.flush()
		self.population = sorted(self.population, key = lambda x: x.fitness, reverse = True)
		self.population = self.population[:int(self.config.POPULATION_SURVIVAL_RATE * self.config.POPULATION_SIZE)]
		self.pop_dict = {}
		for program in self.population:
			if (len(program.vars['global']) not in self.pop_dict):
				self.pop_dict[len(program.vars['global'])] = []
			self.pop_dict[len(program.vars['global'])].append(program)

		for iteration in range(self.config.OPTIM_ITER):
			start = time.time()

			childs = []
			surviving_children_counter = 0
			for l in range(int(self.config.BIRTH_RATE * self.config.POPULATION_SIZE)):
				mother = random.choice(self.population)
				father = random.choice(self.pop_dict[len(mother.vars['global'])])
				child = self.mate(mother, father)
				if (np.isfinite(child.fitness) and child.fitness > self.population[int(self.config.POPULATION_SURVIVAL_RATE * len(self.population))].fitness):
					successive_standstills = 0
					while (successive_standstills <= self.config.MAX_SELF_EVOLUTION_ITER):
						child, is_updated = self.get_evolved_program(child)
						if (is_updated):
							successive_standstills = 0
						else:
							successive_standstills += 1
					if (np.isfinite(program.fitness)):
						surviving_children_counter += 1
						childs.append(child)
			children_survival_rate = surviving_children_counter / int(self.config.BIRTH_RATE * self.config.POPULATION_SIZE)
			self.population.extend(childs)

			surviving_immigrant_counter = 0
			immigrants = []
			for l in range(int(self.config.IMMIGRATION_RATE * self.config.POPULATION_SIZE)):
				immigrant = self.create_random_program()
				if (np.isfinite(immigrant.fitness) and immigrant.fitness > self.population[int(self.config.POPULATION_SURVIVAL_RATE * len(self.population))].fitness):
					successive_standstills = 0
					while (successive_standstills <= self.config.MAX_SELF_EVOLUTION_ITER):
						immigrant, is_updated = self.get_evolved_program(immigrant)
						if (is_updated):
							successive_standstills = 0
						else:
							successive_standstills += 1
					if (np.isfinite(immigrant.fitness)):
						surviving_immigrant_counter += 1
						immigrants.append(immigrant)
			immigrant_survival_rate = surviving_immigrant_counter / int(self.config.IMMIGRATION_RATE * self.config.POPULATION_SIZE)
			self.population.extend(immigrants)

			self.population = sorted(self.population, key = lambda x: x.fitness, reverse = True)
			self.population = self.population[:int(self.config.POPULATION_SURVIVAL_RATE * self.config.POPULATION_SIZE)]
			self.pop_dict = {}
			for program in self.population:
				if (len(program.vars['global']) not in self.pop_dict):
					self.pop_dict[len(program.vars['global'])] = []
				self.pop_dict[len(program.vars['global'])].append(program)

			end = time.time()
			print('Iteration report ' + str(iteration) + '/' + str(self.config.OPTIM_ITER))
			print('{:<30} {:<.2f}'.format('Best fitness', self.population[0].fitness))
			print('{:<30} {:<.2f}'.format('Average fitness', sum(program.fitness for program in self.population) / len(self.population)))
			print('{:<30} {:<.2f}'.format('Average program length', sum(len(program.assignments['preloop']) + len(program.assignments['loop']) + len(program.assignments['postloop']) for program in self.population) / len(self.population)))
			print('{:<30} {:<.2%}'.format('Child survival rate', children_survival_rate))
			print('{:<30} {:<.2%}'.format('Immigrant survival rate', immigrant_survival_rate))
			print('{:<30} {:<.2f}'.format('Elapsed time [s]', end - start))
			print('{:<30} {:<d}'.format('Population size', len(self.population)))
			print('\nBest program: \n')
			print(self.population[0])
			# for k in range(min(10, len(self.population))):
			# 	print('Rank {:<3d}: {:<10.5f}'.format(k + 1, self.population[k].fitness))
			print('\n\n')

		print('\nWinner:\n' + '-' * 40 + '\n')
		print(self.population[0])

		mother = self.population[0]
		father = random.choice(self.pop_dict[len(mother.vars['global'])])
		print('Mother', mother)
		print('Father', father)
		print('Child', self.mate(mother, father))
		# i = 0	
		# version = 0
		# program = self.create_random_program()
		# while True:

				
			# 	if (is_updated):
			# 		version += 1
			# 		print('Iteration', i, ', version', version)
			# 		print(program)
			# 		print('Sum error: ', program.mse)
			# 		print('Fitness:', program.fitness)
			# 	i += 1


			# 	# if (program.fitness > best_fitness):
			# 	# 	print()
			# 	# 	print('Iteration', i, ', version', version)
			# 	# 	print(program)
			# 	# 	print('Sum error: ', program.mse)
			# 	# 	print('Fitness:', program.fitness)
			# 	# 	best_fitness = program.fitness
			# 	# 	version += 1
			# 	# 	self.evolve(program)
			# 	# i += 1

	def create_random_program(self):
		program = Program.Program(self.config)

		n_consts				= abs(int(np.random.normal(		  self.config.N_CONSTS_EV, 							self.config.N_CONSTS_SD)))
		n_global_vars 			= max(1, abs(int(np.random.normal(self.config.N_GLOBAL_VARS_EV, 					self.config.N_GLOBAL_VARS_SD))))
		n_loop_vars 			= max(1, abs(int(np.random.normal(self.config.N_LOOP_VARS_EV, 						self.config.N_LOOP_VARS_SD))))
		n_loop_assignments  	= max(1, abs(int(np.random.normal(self.config.N_LOOP_ASSIGNMENTS_EV + n_loop_vars, 	self.config.N_LOOP_ASSIGNMENTS_SD))))
		n_postloop_assignments 	= max(1, abs(int(np.random.normal(self.config.N_POSTLOOP_ASSIGNMENTS_EV, 			self.config.N_POSTLOOP_ASSIGNMENTS_SD))))

		for i in range(n_consts):
			value = self.create_random_const()
			const = 'const_' + str(i)
			program.consts['loop'].append(const)
			program.consts['global'].append(const)
			assignment = Program.Assignment(self.config, const, False, value)
			program.assignments['preloop'].append(assignment)
		for i in range(n_global_vars):
			var = 'gvar_' + str(i)
			init_var_as_const = np.random.random() < self.config.P_INIT_VAR_AS_CONST
			if (init_var_as_const):
				assignment = self.create_random_assignment('preloop', [], program.consts['global'], False, var)
			else:
				assignment = Program.Assignment(self.config, var, False, 0)
			program.vars['global'].append(var)
			program.assignments['preloop'].append(assignment)
		for i in range(n_loop_assignments):
			var = 'lvar_' + str(np.random.randint(0, n_loop_vars))
			assignment = self.create_random_assignment('loop', program.vars['loop'], program.consts['loop'], False, var)
			program.vars['loop'].append(var)
			program.assignments['loop'].append(assignment)
		for i in range(n_global_vars):
			var = 'gvar_' + str(i)
			assignment = self.create_random_assignment('loop', program.vars['loop'], program.consts['loop'], True, var)
			program.assignments['loop'].append(assignment)
		for i in range(n_postloop_assignments):
			var = 'gvar_' + str(np.random.randint(0, n_global_vars))
			assignment = self.create_random_assignment('postloop', program.vars['global'], program.consts['global'], False, var)
			program.vars['global'].append(var)
			program.assignments['postloop'].append(assignment)
		prediction = Program.Assignment(self.config, 'y_hat', False, 'gvar_' + str(np.random.randint(0, n_global_vars)))
		program.assignments['postloop'].append(prediction)

		program.reduce()

		mse = program.get_mse(self.X_train, self.y_train)
		program.fitness = self.get_fitness(mse, program)
		program.mse = mse

		return program
