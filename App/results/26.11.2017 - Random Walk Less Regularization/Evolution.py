import Program
import Tree
import Common
import numpy as np
import random
import sys
import multiprocessing
import os
import time
import signal
import copy
import matplotlib.pyplot as plt

def sample_categorical_distribution(table, avoid = None):
	r = np.random.random()
	upto = 0

	if (avoid == None):
		for key in table:
			if (upto + table[key] >= r):
				break
			upto += table[key]
		else:
			assert False, 'Should not get here'
		return key
	else:
		successive_failures = 0
		selected = None
		while (selected == None or selected == avoid):
			for key in table:
				if (upto + table[key] >= r):
					break
				upto += table[key]
			else:
				assert False, 'Should not get here'
			selected = key
			successive_failures += 1
			if (successive_failures > 100):
				#print('Failed to sample probability distribution.')
				return False
		return selected

def create_random_const(config):
	is_integer = np.random.random() < config.P_CONST_IS_INTEGER
	const = np.random.normal(config.CONST_EV, config.CONST_SD)
	if (is_integer):
		const = int(const)
	return const

def create_random_program(config):
	program = Program.Program('Alien', config)

	n_consts = abs(int(np.random.normal(config.N_CONSTS_EV, config.N_CONSTS_SD)))
	for i in range(n_consts):
		value = create_random_const(config)
		const = 'const_' + str(random.getrandbits(100))
		program.consts['loop'].append(const)
		program.consts['global'].append(const)
		assignment = Program.Assignment(config, const, False, value)
		program.assignments['preloop'].append(assignment)

	program.loop_tree = rec_create_random_tree(config, program.consts['loop'])
	program.rec_reduce()

	return program

def rec_create_random_tree(config, consts, height = 0, target_height = None, first_caller = True):
	if (first_caller):
		target_height = random.randint(1, config.MAX_INIT_TREE_HEIGHT)

	operator = sample_categorical_distribution(config.BINARY_OPERATIONS_PROBABILITIES)

	is_transformed = random.random() < config.P_USE_UNARY_TRANSFORMATION
	if (is_transformed):
		transformation = sample_categorical_distribution(config.UNARY_OPERATIONS_PROBABILITIES)
		root = Tree.Node(operator, None, None, transformation)
	else:
		root = Tree.Node(operator, None, None)

	is_terminal = (height >= target_height) or (height > 0 and random.random() < config.P_CHOOSE_TERMINAL)

	if (is_terminal):
		is_const = random.random() < config.P_CHOOSE_CONST
		is_transformed = random.random() < config.P_USE_UNARY_TRANSFORMATION
		if (is_const):
			var = random.choice(consts)
		else:
			var = 'X[i][' + str(np.random.randint(0, config.K)) + ']'
		if (is_transformed):
			transformation = sample_categorical_distribution(config.UNARY_OPERATIONS_PROBABILITIES)
			return transformation + '(' + var + ')'
		else:
			return var
	else:
		root.left_child = rec_create_random_tree(config, consts, height + 1, target_height, False)
		root.right_child = rec_create_random_tree(config, consts, height + 1, target_height, False)

	return root

def rec_select_random_tree_node(node, cur_level = 1, target_level = None, first_caller = True):
	options = []
	if (target_level == None):
		target_level = random.randint(1, node.rec_get_height())
		#print('Selected target level: ', target_level, '\t\t')
	if (cur_level == target_level):
		#print('Selectable node: ', node)
		options.append(node)
	else:
		if (type(node.left_child) is Tree.Node):
			options.extend(rec_select_random_tree_node(node.left_child, cur_level + 1, target_level, False))
		if (type(node.right_child) is Tree.Node):
			options.extend(rec_select_random_tree_node(node.right_child, cur_level + 1, target_level, False))
	if (first_caller):
		return random.choice(options)
	else:
		return options

class IslandGenerationData:
	pass

class Island:
	Counter = 1
	def __init__(self, root):
		self.population = []
		self.distribution = None
		self.history = []
		self.id = 'Island_' + str(Island.Counter)
		Island.Counter += 1
		self.report_path = os.path.normpath(root + '/report/islands/' + self.id)
		os.makedirs(self.report_path)

	def update_distribution(self):
		table = {}
		for program in self.population:
			table[program] = np.exp(np.exp(program.fitness)) - np.exp(1)
		self.distribution = Common.get_prob_distribution_from_rel_frequencies(table)

	def save_report(self):
		best_fitness_arr = []
		avg_fitness_arr = []
		for generation_data in self.history:
			best_fitness_arr.append(generation_data.best_fitness)
			avg_fitness_arr.append(generation_data.avg_fitness)

		type_pie_chart_fig = plt.figure(figsize=(12.0, 12.0))
		labels = []
		fracs = []
		for label in self.history[-1].program_type_distribution:
			labels.append(label)
			fracs.append(self.history[-1].program_type_distribution[label])
		plt.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)
		type_pie_chart_fig.savefig(os.path.normpath(self.report_path + '/program_types.png'))

		best_fitness_fig = plt.figure(figsize=(12.0, 7.0))
		plt.plot(best_fitness_arr)
		plt.plot(avg_fitness_arr)
		plt.title(self.id + ' - Fitness plot')
		plt.ylabel('Fitness')
		plt.xlabel('Generation')
		plt.legend(['Best', 'Average'])
		best_fitness_fig.savefig(os.path.normpath(self.report_path + '/fitness.png'))
		plt.close()

class EvolutionGenerationData:
	pass

class Evolution():
	def __init__(self, root, config, X_train, y_train, X_val, y_val):
		self.root 			= root
		self.report_path 	= os.path.normpath(root + '/report')
		self.config 		= config
		self.X_train 		= X_train
		self.y_train 		= y_train	
		self.var_y_train 	= y_train.var()
		self.X_val 			= X_val
		self.y_val 			= y_val
		self.var_y_val 		= y_val.var()
		self.islands 		= []
		self.history 		= []
		for p in range(self.config.NUM_ISLANDS):
			self.islands.append(Island(root = root))

		try:
			self.optimize()
			self.on_finished()
		except (KeyboardInterrupt, SystemExit):
			print('Encountered KeyboardInterrupt or SystemExit exception.')
			self.on_finished()

	def on_finished(self):
		print('Evolution is finished.')
		self.save_report()
		for island in self.islands:
			island.save_report()

	def mate(self, mother, father):
		child = Program.Program('Child', self.config)
		child.assignments['preloop'] = []
		for assignment in mother.assignments['preloop']:
			new_assignment = Program.Assignment(self.config, assignment.variable, False, np.random.normal(assignment.lhs, self.config.CONST_MATING_DRIFT*abs(assignment.lhs)))
			child.assignments['preloop'].append(new_assignment)
			if ('const' in assignment.variable):
				child.consts['loop'].append(assignment.variable)
				child.consts['global'].append(assignment.variable)
		for assignment in father.assignments['preloop']:
			new_assignment = Program.Assignment(self.config, assignment.variable, False, np.random.normal(assignment.lhs, self.config.CONST_MATING_DRIFT*abs(assignment.lhs)))
			child.assignments['preloop'].append(new_assignment)
			if ('const' in assignment.variable):
				child.consts['loop'].append(assignment.variable)
				child.consts['global'].append(assignment.variable)

		tree_mother = copy.deepcopy(mother.loop_tree)
		tree_father = copy.deepcopy(father.loop_tree)
		crossover_mother = rec_select_random_tree_node(tree_mother)
		crossover_father = rec_select_random_tree_node(tree_father)

		crossover_direction = random.choice(['LEFT', 'RIGHT'])
		if (crossover_direction == 'LEFT'):
			crossover_mother.left_child = crossover_father
		else:
			crossover_mother.right_child = crossover_father

		child.loop_tree = tree_mother
		child.rec_reduce()

		return child

	def mutate(self, mother):
		child = Program.Program('Mutant', self.config)
		child.assignments['preloop'] = []
		for assignment in mother.assignments['preloop']:
			new_assignment = Program.Assignment(self.config, assignment.variable, False, np.random.normal(assignment.lhs, self.config.CONST_MATING_DRIFT*abs(assignment.lhs)))
			child.assignments['preloop'].append(new_assignment)
			if ('const' in assignment.variable):
				child.consts['loop'].append(assignment.variable)
				child.consts['global'].append(assignment.variable)

		tree_mother = copy.deepcopy(mother.loop_tree)
		crossover_mother = rec_select_random_tree_node(tree_mother)

		crossover_direction = random.choice(['LEFT', 'RIGHT'])
		if (crossover_direction == 'LEFT'):
			crossover_mother.left_child = rec_create_random_tree(self.config, child.consts['loop'])
		else:
			crossover_mother.right_child = rec_create_random_tree(self.config, child.consts['loop'])

		child.loop_tree = tree_mother
		child.rec_reduce()

		return child

	def get_fitness(self, program, dataset, use_batch = True):
		assert(dataset in ['TRAINING', 'VALIDATION'])
		if (dataset == 'TRAINING'):
			mse 			= program.get_mse(self.X_train, self.y_train, use_batch)
			r_squared	   	= 1 - mse / self.var_y_train
		if (dataset == 'VALIDATION'):
			mse 			= program.get_mse(self.X_val, self.y_val, use_batch)
			r_squared	   	= 1 - mse / self.var_y_val
		tree_size 		= program.loop_tree.rec_get_node_count()
		tree_height 	= program.loop_tree.rec_get_height()
		correctness    = 1 if mse == 0 else 0
		z = self.config.FITNESS_WEIGHTS['intercept'] 							+ \
			self.config.FITNESS_WEIGHTS['r_squared'] 		* r_squared 		+ \
		    self.config.FITNESS_WEIGHTS['tree_size'] 		* tree_size 		+ \
		    self.config.FITNESS_WEIGHTS['tree_height'] 		* tree_height   	+ \
		    self.config.FITNESS_WEIGHTS['correctness'] 		* correctness
		fitness = 1/(1 + np.exp(-z))
		return fitness

	def on_islands_changed(self, res):
		self.islands = res

	def initiate_island(self, island):
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		try:
			for i in range(self.config.POPULATION_SIZE):
				sys.stdout.write('Creating program ' + str(i + 1) + ' / ' + str(self.config.POPULATION_SIZE) + '...\r')
				sys.stdout.flush()
				program = create_random_program(self.config)
				program.fitness = self.get_fitness(program, 'TRAINING')
				if (np.isfinite(program.fitness)):
					successive_standstills = 0
					island.population.append(program)
			sys.stdout.write('\t\t\t\t\t\t\r')
			sys.stdout.flush()
			island.population = sorted(island.population, key = lambda x: x.fitness, reverse = True)
			island.population = island.population[:int(self.config.POPULATION_SURVIVAL_RATE * self.config.POPULATION_SIZE)]
			island.update_distribution()
			return island
		except (KeyboardInterrupt, SystemExit):
			return	

	def update_island(self, p):
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		try:
			island = self.islands[p]

			# ----------------------
			#    C H I L D R E N 
			# ----------------------

			childs = []
			surviving_children_counter = 0
			num_mating_processes = int(self.config.BIRTH_RATE * self.config.POPULATION_SIZE)
			for l in range(num_mating_processes):
				sys.stdout.write('Performing mating ' + str(l + 1) + ' / ' + str(num_mating_processes) + '...\r')
				sys.stdout.flush()

				mother = sample_categorical_distribution(island.distribution)
				father = None
				father = sample_categorical_distribution(island.distribution, mother)
				if (father == False):
					continue

				for i in range(self.config.NUM_MATING_ITERATIONS):
					child = self.mate(mother, father)
					child.fitness = self.get_fitness(child, 'TRAINING')
					if (np.isfinite(child.fitness) and child.fitness > island.population[int(self.config.POPULATION_SURVIVAL_RATE * len(island.population))].fitness):
						surviving_children_counter += 1
						childs.append(child)

			child_survival_rate = surviving_children_counter / num_mating_processes if num_mating_processes > 0 else 0
			island.population.extend(childs)

			sys.stdout.write('\t\t\t\t\t\t\r')
			sys.stdout.flush()

			# ----------------------
			#  I M M I G R A N T S
			# ----------------------

			surviving_immigrant_counter = 0
			immigrants = []
			num_immigrations = int(self.config.MIGRATION_RATE * self.config.POPULATION_SIZE)
			immigrant_ids = []
			for l in range(num_immigrations):
				sys.stdout.write('Performing immigration ' + str(l + 1) + ' / ' + str(num_immigrations) + '...\r')
				sys.stdout.flush()
				origin = None
				while (origin == None or p == origin):
					origin = random.randint(0, self.config.NUM_ISLANDS - 1)
				immigrant_id = None
				while (immigrant_id == None or immigrant_id in immigrant_ids):
					immigrant = copy.deepcopy(sample_categorical_distribution(self.islands[origin].distribution))
					immigrant_id = immigrant.id
				if (np.isfinite(immigrant.fitness) and immigrant.fitness > island.population[int(self.config.POPULATION_SURVIVAL_RATE * len(island.population))].fitness):
					surviving_immigrant_counter += 1
					immigrant.type = 'Immigrant'
					immigrants.append(immigrant)
					immigrant_ids.append(immigrant.id)

			immigrant_survival_rate = surviving_immigrant_counter / num_immigrations if num_immigrations > 0 else 0
			island.population.extend(immigrants)

			sys.stdout.write('\t\t\t\t\t\t\r')
			sys.stdout.flush()

			# ----------------------
			#      M U T A N T S 
			# ----------------------

			mutants = []
			surviving_mutants_counter = 0
			num_mutation_processes = int(self.config.MUTATION_RATE * self.config.POPULATION_SIZE)
			for l in range(num_mutation_processes):
				sys.stdout.write('Performing mutation ' + str(l + 1) + ' / ' + str(num_mutation_processes) + '...\r')
				sys.stdout.flush()

				mother = sample_categorical_distribution(island.distribution)
				mutant = self.mutate(mother)
				mutant.fitness = self.get_fitness(mutant, 'TRAINING')

				if (np.isfinite(mutant.fitness) and mutant.fitness > island.population[int(self.config.POPULATION_SURVIVAL_RATE * len(island.population))].fitness):
					surviving_mutants_counter += 1
					mutants.append(mutant)

			mutant_survival_rate = surviving_mutants_counter / num_mutation_processes if num_mutation_processes > 0 else 0
			island.population.extend(mutants)

			sys.stdout.write('\t\t\t\t\t\t\r')
			sys.stdout.flush()

			# ----------------------
			#     A L I E N S
			# ----------------------

			surviving_aliens_counter = 0
			aliens = []
			num_aliens = int(self.config.ALIEN_ARRIVAL_RATE * self.config.POPULATION_SIZE)
			for l in range(num_aliens):
				sys.stdout.write('Performing alien arrival ' + str(l + 1) + ' / ' + str(num_aliens) + '...\r')
				sys.stdout.flush()

				alien = create_random_program(self.config)
				alien.fitness = self.get_fitness(alien, 'TRAINING')
				if (np.isfinite(alien.fitness) and alien.fitness > island.population[int(self.config.POPULATION_SURVIVAL_RATE * len(island.population))].fitness):
					surviving_aliens_counter += 1
					aliens.append(alien)
			alien_survival_rate = surviving_aliens_counter / num_aliens if num_aliens > 0 else 0
			island.population.extend(aliens)

			sys.stdout.write('\t\t\t\t\t\t\r')
			sys.stdout.flush()

			# -----------------------------------------------
			#     S U R V I V A L  O F  T H E  F I T T E S T
			# -----------------------------------------------

			island.population = sorted(island.population, key = lambda x: x.fitness, reverse = True)
			island.population = island.population[:int(self.config.POPULATION_SURVIVAL_RATE * self.config.POPULATION_SIZE)]
			island.update_distribution()

			generation_data = IslandGenerationData()
			generation_data.best_code 					= str(island.population[0])
			generation_data.best_fitness 				= island.population[0].fitness
			generation_data.avg_fitness 					= sum(program.fitness for program in island.population) / len(island.population)
			generation_data.program_type_distribution 	= {}
			for program in island.population:
				if (program.type not in generation_data.program_type_distribution):
					generation_data.program_type_distribution[program.type] = 0
				generation_data.program_type_distribution[program.type] += 1
			generation_data.num_survived_children 		= surviving_children_counter
			generation_data.num_survived_immigrants 	= surviving_immigrant_counter
			generation_data.num_survived_mutants 		= surviving_mutants_counter
			generation_data.num_survived_aliens 		= surviving_aliens_counter	
			generation_data.child_survival_rate 		= child_survival_rate
			generation_data.immigrant_survival_rate 	= immigrant_survival_rate
			generation_data.mutant_survival_rate 		= mutant_survival_rate
			generation_data.alien_survival_rate 		= alien_survival_rate
			generation_data.population_size 			= len(island.population)

			island.history.append(generation_data)

			return island

		except (KeyboardInterrupt, SystemExit):
			return	

	def optimize(self):
		pool = multiprocessing.Pool(self.config.NUM_POOLS)
		try:
			res = pool.map_async(self.initiate_island, [self.islands[p] for p in range(self.config.NUM_ISLANDS)], callback = self.on_islands_changed)
			#res.get(60)
		except KeyboardInterrupt as e:
			print("Caught KeyboardInterrupt, terminating workers")
			pool.terminate()
			raise e
		else:
			pool.close()
			pool.join()

		for iteration in range(self.config.OPTIM_ITER):
			start = time.time()
			pool = multiprocessing.Pool(self.config.NUM_POOLS)
			try:
				res = pool.map_async(self.update_island, [p for p in range(self.config.NUM_ISLANDS)], callback = self.on_islands_changed)
				#res.get(60)
			except KeyboardInterrupt as e:
				print("Caught KeyboardInterrupt, terminating workers")
				pool.terminate()
				raise e
			else:
				pool.close()
				pool.join()

			self.update_history()
			end = time.time()
			print('Generation {:<d} elapsed time [s] {:<.2f}'.format(iteration, end - start))

		print('Finished evolution.')

	def update_history(self):
		print('Updating history...')

		best_program = None
		best_fitness = 0
		best_island = None
		for island in self.islands:
			if (island.population[0].fitness > best_fitness):
				best_program = island.population[0]
				best_fitness = island.population[0].fitness
				best_island = island.id

		with open(os.path.normpath(self.report_path + '/best_program.txt'), 'w') as file:
			file.write('---Best program---\n')
			file.write('Fitness: ' + str(best_fitness) + '\n')
			file.write(best_island + '\n\n')
			file.write(str(best_program))

		generation_data = EvolutionGenerationData()
		generation_data.best_island 		= best_island
		generation_data.best_code 			= str(best_program)
		generation_data.best_train_fitness 	= self.get_fitness(best_program, 'TRAINING', False)
		generation_data.best_val_fitness 	= self.get_fitness(best_program, 'VALIDATION', False)

		self.history.append(generation_data)

	def save_report(self):
		best_fitness_fig = plt.figure(figsize=(12.0, 7.0))
		labels = []
		for island in self.islands:
			labels.append(island.id)
			best_fitness_arr = []
			for generation_data in island.history:
				best_fitness_arr.append(generation_data.best_fitness)
			plt.plot(best_fitness_arr)
		plt.title('Fitness plot')
		plt.ylabel('Fitness score')
		plt.xlabel('Generation')
		plt.legend(labels)
		best_fitness_fig.savefig(os.path.normpath(self.report_path + '/fitness.png'))

		best_program = None
		best_fitness = 0
		best_island = None
		for island in self.islands:
			if (island.population[0].fitness > best_fitness):
				best_program = island.population[0]
				best_fitness = island.population[0].fitness
				best_island = island.id

		with open(os.path.normpath(self.report_path + '/best_program.txt'), 'w') as file:
			file.write('---Best program---\n')
			file.write('Fitness score: ' + str(best_fitness) + '\n')
			file.write(best_island + '\n\n')
			file.write(str(best_program))

		with open(os.path.normpath(self.report_path + '/code_history.txt'), 'w') as file:
			for generation in range(len(self.history)):
				generation_data = self.history[generation]
				file.write('-' * 20 + ' Generation ' + str(generation + 1) + ' ' + '-' * 20 +  '\n')
				file.write('Fitness score (Training): ' + str(generation_data.best_train_fitness) + '\n')
				file.write('Fitness score (Validation): ' + str(generation_data.best_val_fitness) + '\n')
				file.write('Island: ' + generation_data.best_island + '\n')
				file.write(generation_data.best_code)
				file.write('\n\n\n')

		validation_curve_fig = plt.figure(figsize=(12.0, 7.0))
		labels = ['Training score', 'Validation score']
		best_train_fitness_arr = []
		best_val_fitness_arr = []
		for generation_data in self.history:
			best_train_fitness_arr.append(generation_data.best_train_fitness)
			best_val_fitness_arr.append(generation_data.best_val_fitness)
		plt.plot(best_train_fitness_arr)
		plt.plot(best_val_fitness_arr)
		plt.title('Validation curve')
		plt.ylabel('Fitness score')
		plt.xlabel('Generation')
		plt.legend(labels)
		validation_curve_fig.savefig(os.path.normpath(self.report_path + '/validation_curve.png'))

		plt.close()