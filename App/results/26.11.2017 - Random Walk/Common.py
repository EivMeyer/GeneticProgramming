def get_prob_distribution_from_rel_frequencies(table):
	distr = {}
	total_frequency = sum(table[key] for key in table)
	for key in table:
		distr[key] = table[key] / total_frequency
	return distr