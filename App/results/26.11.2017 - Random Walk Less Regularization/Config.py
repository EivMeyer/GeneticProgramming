import Common
import multiprocessing

class Configuration:
	def __init__(self):
		# Time series vector dimension
		self.K = 1

		# Set of binary operators usable by a program (operator -> relative frequency)
		self.BINARY_OPERATIONS = {
			'np.add': 			50,
			'np.subtract': 		25,
			'np.multiply': 		15,
			'np.divide': 		10,
			'np.power': 		2,
			'np.mod': 			0.2,
			'np.fmin': 			0.2,
			'np.fmax': 			0.2,
			'np.greater': 		0.1, 
			'np.greater_equal': 0.1, 
			'np.less': 			0.1, 
			'np.less_equal': 	0.1, 
			'np.equal': 		0.01, 
			'np.not_equal': 	0.01
		}

		# Set of unary operators usable by a program (operator -> relative frequency)
		self.UNARY_OPERATIONS = {    		
			'np.exp': 			4, 		
			#'np.log': 			4, 		
			#'np.sqrt': 			4, 		
			#'np.log2': 			2, 			
			'np.sin': 			1, 		
			'np.cos': 			1, 		
			'np.tan': 			0.6, 		
			'np.sign': 			0.4, 
			'np.fabs': 			0.1,		
			'np.floor': 		0.02, 		
			'np.ceil': 			0.02, 		
			'np.remainder': 	0.01,
		}

		self.BINARY_OPERATIONS_PROBABILITIES = Common.get_prob_distribution_from_rel_frequencies(self.BINARY_OPERATIONS)
		self.UNARY_OPERATIONS_PROBABILITIES = Common.get_prob_distribution_from_rel_frequencies(self.UNARY_OPERATIONS)

		# Parameters for random program generation
		self.N_CONSTS_EV 				= 2
		self.N_CONSTS_SD 				= 2
		self.CONST_EV 					= 5
		self.CONST_SD 					= 5
		self.MAX_INIT_TREE_HEIGHT 		= 10
		self.P_CONST_IS_INTEGER 		= 0
		self.P_CHOOSE_CONST 			= 0.3
		self.P_CHOOSE_TERMINAL 			= 0.25
		self.P_USE_UNARY_TRANSFORMATION = 0.25
		self.CONST_MATING_DRIFT 		= 1/20

		assert(self.P_CONST_IS_INTEGER 			>= 0 and self.P_CONST_IS_INTEGER 	  		<= 1)
		assert(self.P_CHOOSE_CONST 				>= 0 and self.P_CHOOSE_CONST 	      		<= 1)
		assert(self.P_CHOOSE_TERMINAL 			>= 0 and self.P_CHOOSE_TERMINAL 	  		<= 1)
		assert(self.P_USE_UNARY_TRANSFORMATION 	>= 0 and self.P_USE_UNARY_TRANSFORMATION  	<= 1)

		# Parameters for fitness function
		self.FITNESS_WEIGHTS = {
			'intercept': 			-1,
			'correctness': 			3,
			'r_squared': 			1,
			'tree_size': 			-0.000005,
			'tree_height': 			-0.00005
		}

		# Evolution parameters
		self.BATCH_SIZE 				= 0.1
		self.OPTIM_ITER 				= 50
		self.POPULATION_SIZE 			= 1000
		self.POPULATION_SURVIVAL_RATE 	= 0.5
		self.NUM_ISLANDS 				= 4
		self.MIGRATION_RATE 			= 0.02 * (0 if self.NUM_ISLANDS <= 1 else 1)
		self.ALIEN_ARRIVAL_RATE 		= 0.2
		self.NUM_MATING_ITERATIONS 		= 1
		self.BIRTH_RATE 				= 1
		self.MUTATION_RATE 				= 0.5
		self.MAX_SELF_EVOLUTION_ITER 	= -1

		# Hardware
		self.NUM_POOLS 					= min(self.NUM_ISLANDS, multiprocessing.cpu_count())

	def __str__(self):
		output = ('{:^30}\n{:^30}\n').format('Operator Probabilities:', '-'*30)
		for operator in self.OPERATOR_PROBABILITIES:
			output += ' {:<20}{:<.3%}\n'.format(operator, self.OPERATOR_PROBABILITIES[operator])
		return output