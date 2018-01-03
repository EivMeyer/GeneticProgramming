class Configuration:
	def __init__(self):
		# Time series vector dimension
		self.K = 1

		# Set of operators usable by a program (frequency, parameters)
		self.OPERATORS = {              	
			'': 				(50, 		1),
			'np.add': 			(50, 		2),
			'np.subtract': 		(25, 		2), 
			'np.multiply': 		(15, 		2), 
			'np.divide': 		(10, 		2),
			'np.exp': 			(10, 		1),
			'np.log': 			(2, 		1),
			'np.sin': 			(1, 		1),
			'np.floor': 		(1, 		1),
			'np.ceil': 			(1, 		1),
			'np.log2': 			(1, 		1),
			'np.power': 		(1, 		2),
			'np.mod': 			(1, 		2),
			'np.sqrt': 			(1, 		1),
			'np.tan': 			(1, 		1),
			'np.cos': 			(1, 		1),
			'np.fabs': 			(1, 		1),
			'np.sign': 			(1, 		1),
			'np.greater': 		(1, 		2),
			'np.greater_equal': (1, 		2),
			'np.less': 			(1, 		2),
			'np.less_equal': 	(1, 		2),
			'np.equal': 		(1, 		2),
			'np.not_equal': 	(1, 		2),
			'np.fmin': 			(0.2, 		2),
			'np.fmax': 			(0.2, 		2),
			'np.remainder': 	(0.1, 		1),
		}

		self.OPERATOR_PROBABILITIES = {}
		total_frequency = sum(self.OPERATORS[operator][0] for operator in self.OPERATORS)
		for operator in self.OPERATORS:
			self.OPERATOR_PROBABILITIES[operator] = self.OPERATORS[operator][0] / total_frequency

		# Parameters for random program generation
		self.N_CONSTS_EV 				= 2
		self.N_CONSTS_SD 				= 2
		self.N_GLOBAL_VARS_EV 			= 50
		self.N_GLOBAL_VARS_SD 			= 2
		self.N_LOOP_VARS_EV 			= 5
		self.N_LOOP_VARS_SD 			= 2
		self.P_INIT_VAR_AS_CONST 		= 0.5 
		self.N_LOOP_ASSIGNMENTS_EV  	= 5
		self.N_LOOP_ASSIGNMENTS_SD 		= 2
		self.N_POSTLOOP_ASSIGNMENTS_EV 	= 50
		self.N_POSTLOOP_ASSIGNMENTS_SD 	= 1
		self.CONST_EV 					= 5
		self.CONST_SD 					= 5
		self.P_CONST_IS_INTEGER 		= 0.5
		self.P_CHOOSE_CONST 			= 0.3
		self.P_CHOOSE_X 				= 0.6

		assert(self.P_INIT_VAR_AS_CONST >= 0 and self.P_INIT_VAR_AS_CONST <= 1)
		assert(self.P_CHOOSE_CONST 		>= 0 and self.P_CHOOSE_CONST 	  <= 1)
		assert(self.P_CHOOSE_X 			>= 0 and self.P_CHOOSE_X 		  <= 1)
		assert(self.P_CHOOSE_X 				   + self.P_CHOOSE_CONST 	  <= 1)

		# Parameters for fitness function
		self.FITNESS_WEIGHTS = {
			'correctness': 		3,
			'r_squared': 		3,
			'variable_count': 	-0.1,
			'program_length': 	-0.1
		}

		# Evolution parameters
		self.BATCH_SIZE 				= 1
		self.OPTIM_ITER 				= 100
		self.POPULATION_SIZE 			= 10000
		self.POPULATION_SURVIVAL_RATE 	= 0.5
		self.BIRTH_RATE 				= 1
		self.IMMIGRATION_RATE 			= 0.2
		self.MAX_SELF_EVOLUTION_ITER 	= -1
		self.EVOLUTION_SPEED 			= 5
		self.P_DELETE_LINE 				= 0.02 	* self.EVOLUTION_SPEED
		self.P_MODIFY_LINE 				= 0.1	* self.EVOLUTION_SPEED
		self.P_ADD_LINE 				= 0.03 	* self.EVOLUTION_SPEED

	def __str__(self):
		output = ('{:^30}\n{:^30}\n').format('Operator Probabilities:', '-'*30)
		for operator in self.OPERATOR_PROBABILITIES:
			output += ' {:<20}{:<.3%}\n'.format(operator, self.OPERATOR_PROBABILITIES[operator])
		return output