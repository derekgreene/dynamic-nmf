# --------------------------------------------------------------

class ModelSimilarity:
	''' 
	Uses a model (e.g. Word2Vec model) to calculate the similarity between two terms.
	'''
	def __init__( self, model ):
		self.model = model	

	def similarity( self, ranking_i, ranking_j ):
		sim = 0.0
		pairs = 0
		for term_i in ranking_i:
			for term_j in ranking_j:
				try:
					sim += self.model.similarity(term_i, term_j)
					pairs += 1
				except:
					#print "Failed pair (%s,%s)" % (term_i,term_j)
					pass
		if pairs == 0:
			return 0.0
		return sim/pairs


# --------------------------------------------------------------

class WithinTopicMeasure:
	'''
	Measures within-topic coherence for a topic model, based on a set of term rankings.
	'''
	def __init__( self, metric ):
		self.metric = metric

	def evaluate_ranking( self, term_ranking ):
		return self.metric.similarity( term_ranking, term_ranking ) 

	def evaluate_rankings( self, term_rankings ):
		scores = []
		overall = 0.0
		for topic_index in range(len(term_rankings)):
			score = self.evaluate_ranking( term_rankings[topic_index] ) 
			scores.append( score )
			overall += score
		overall /= len(term_rankings)
		return overall
