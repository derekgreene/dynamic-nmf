#!/usr/bin/env python
"""
Tool to generate a dynamic topic model, by combining a set of time window topic models.

Sample usage:

python find-dynamic-topics.py out/month1_windowtopics_k05.pkl out/month2_windowtopics_k08.pkl out/month3_windowtopics_k08.pkl -k 4,10 -o out -m out/w2v-model.bin 
"""
import os, sys, random, operator
import logging as log
from optparse import OptionParser
import numpy as np
import sklearn.preprocessing
import text.util
import unsupervised.nmf, unsupervised.rankings, unsupervised.coherence

# --------------------------------------------------------------

class TopicCollection:

	def __init__( self, top_terms = 0, threshold = 1e-6 ):
		# settings
		self.top_terms = top_terms
		self.threshold = threshold
		# state
		self.topic_ids = []		
		self.all_weights = []
		self.all_terms = set()		

	def add_topic_model( self, H, terms, window_topic_labels ):
		'''
		Add topics from a window topic model to the collection.
		'''
		k = H.shape[0]
		for topic_index in range(k):
			topic_weights = {}
			# use top terms only (sparse topic representation)?
			if self.top_terms > 0:
				top_indices = np.argsort( H[topic_index,:] )[::-1]
				for term_index in top_indices[0:self.top_terms]:
					topic_weights[terms[term_index]] = H[topic_index,term_index]
					self.all_terms.add( terms[term_index] )
			# use dense window topic vectors
			else:
				total_weight = 0.0
				for term_index in range(len(terms)):
					total_weight += H[topic_index,term_index]
				for term_index in range(len(terms)):
					w = H[topic_index,term_index] / total_weight
					if w >= self.threshold:
						topic_weights[terms[term_index]] = H[topic_index,term_index]
						self.all_terms.add( terms[term_index] )
			self.all_weights.append( topic_weights )
			self.topic_ids.append( window_topic_labels[topic_index] )

	def create_matrix( self ):
		'''
		Create the topic-term matrix from all window topics that have been added so far.
		'''
		# map terms to column indices
		all_terms = list(self.all_terms)
		M = np.zeros( (len(self.all_weights), len(all_terms)) )
		term_col_map = {}
		for term in all_terms:
			term_col_map[term] = len(term_col_map)
		# populate the matrix in row-order
		row = 0
		for topic_weights in self.all_weights:
			for term in topic_weights.keys():
				M[row,term_col_map[term]] = topic_weights[term]
			row +=1
		# normalize the matrix rows to L2 unit length
		normalizer = sklearn.preprocessing.Normalizer(norm='l2', copy=True)
		normalizer.fit(M)
		M = normalizer.transform(M)
		return (M,all_terms)

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] window_topics1 window_topics2...")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("-k", action="store", type="string", dest="krange", help="number of topics", default=None)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=200)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", help="path to Word2Vec model, if performing automatic selection of number of topics", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to use, if performing automatic selection of number of topics", default=20)
	parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="display topic descriptors")
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least two window topic files" )
	log.basicConfig(level=20, format='%(message)s')

	# Parse user-specified range for number of topics K
	if options.krange is None:
		parser.error("Must specific number of topics, or a range for the number of topics")
	kparts = options.krange.split(",")
	kmin = int(kparts[0])

	# Set random state
	random_seed = options.seed
	if random_seed < 0:
		random_seed = random.randint(1,100000)
	np.random.seed( random_seed )
	random.seed( random_seed )			
	log.info("Using random seed %s" % random_seed )

	# Output directory for results
	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Will we use automatic model selection?
	validation_measure = None
	if len(kparts) == 1:
		kmax = kmin
	else:
		kmax = int(kparts[1])
		# any word2vec model specified?
		if not options.model_path is None:
			log.info( "Loading Word2Vec model from %s ..." % options.model_path )
			import gensim
			model = gensim.models.Word2Vec.load(options.model_path) 
			validation_measure = unsupervised.coherence.WithinTopicMeasure( unsupervised.coherence.ModelSimilarity(model) )

	# Process each specified window topic model
	log.info("- Processing individual time window topic models ...")
	collection = TopicCollection()
	for window_model_path in args:
		# Load the cached time window
		window_name = os.path.splitext( os.path.split( window_model_path )[-1] )[0]
		(doc_ids, terms, term_rankings, partition, W, H, window_topic_labels) = unsupervised.nmf.load_nmf_results( window_model_path )
		log.info("Loaded %d time window topics from %s" % (len(term_rankings),window_model_path) )
		collection.add_topic_model( H, terms, window_topic_labels )

	# Create the topic-term matrix
	M, all_terms = collection.create_matrix()
	log.info( "Created topic-term matrix of size %dx%d" % M.shape )
	log.debug( "Matrix stats: range=[%.3f,%.3f] mean=%.3f" % ( np.min(M), np.max(M), np.mean(M) ) )

	# NMF implementation
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "nndsvd", random_seed = random_seed )

	# Generate window topic model for the specified range of numbers of topics
	coherence_scores = {}
	for k in range(kmin,kmax+1):
		log.info( "Applying dynamic topic modeling to matrix for k=%d topics ..." % k )
		impl.apply( M, k )
		log.info( "Generated %dx%d factor W and %dx%d factor H" % ( impl.W.shape[0], impl.W.shape[1], impl.H.shape[0], impl.H.shape[1] ) )
		# Create a disjoint partition of documents
		partition = impl.generate_partition()
		# Create topic labels
		topic_labels = []
		for i in range( k ):
			topic_labels.append( "D%02d" % (i+1) )
		# Create term rankings for each topic
		term_rankings = []
		for topic_index in range(k):		
			ranked_term_indices = impl.rank_terms( topic_index )
			term_ranking = [all_terms[i] for i in ranked_term_indices]
			term_rankings.append(term_ranking)
		# Print out the top terms?
		if options.verbose:
			log.info( unsupervised.rankings.format_term_rankings( term_rankings, top = options.top ) )
		# Evaluate topic coherence of this topic model?
		if not validation_measure is None:
			truncated_term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
			coherence_scores[k] = validation_measure.evaluate_rankings( truncated_term_rankings )
			log.info("Model coherence (k=%d) = %.4f" % (k,coherence_scores[k]) )
		# Write results
		results_out_path = os.path.join( dir_out, "dynamictopics_k%02d.pkl"  % (k) )
		unsupervised.nmf.save_nmf_results( results_out_path, collection.topic_ids, all_terms, term_rankings, partition, impl.W, impl.H, topic_labels )

	# Need to select value of k?
	if len(coherence_scores) > 0:
		sx = sorted(coherence_scores.items(), key=operator.itemgetter(1))
		sx.reverse()
		top_k = [ p[0] for p in sx ][0:min(3,len(sx))]
		log.info("- Top recommendations for number of dynamic topics: %s" % ",".join(map(str, top_k)) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
