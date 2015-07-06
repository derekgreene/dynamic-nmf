#!/usr/bin/env python
"""
Tool to generate a dynamic topic model, by combining a set of time window topic models.
"""
import os, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
import sklearn.preprocessing
import text.util, unsupervised.nmf, unsupervised.rankings, unsupervised.util

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
	parser.add_option("-k", action="store", type="int", dest="k", help="number of topics", default=5)
	parser.add_option("-d", "--dims", action="store", type="int", dest="dimensions", help="number of dimensions (top terms) to use", default=20)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=200)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to display when showing results", default=10)
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least two window topic files" )
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Set random state
	np.random.seed( options.seed )
	random.seed( options.seed )	

	# Process each specified window topic model
	log.info("- Processing individual time window topic models ...")
	collection = TopicCollection()
	for window_model_path in args:
		# Load the cached time window
		window_name = os.path.splitext( os.path.split( window_model_path )[-1] )[0]
		(term_rankings, partition, W, H, terms, window_topic_labels) = unsupervised.util.load_nmf_results( window_model_path )
		log.info("Loaded %d time window topics from %s" % (len(term_rankings),window_model_path) )
		collection.add_topic_model( H, terms, window_topic_labels )

	# Create the topic-term matrix
	M, all_terms = collection.create_matrix()
	log.info( "Created topic-term matrix of size %dx%d" % M.shape )
	log.debug( "Matrix stats: range=[%.2f,%.2f] mean=%.2f" % ( np.min(M), np.mean(M), np.max(M) ) )	

	# Generate dynamic NMF topic model for the specified numbers of topics
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "nndsvd" )
	k = options.k
	log.info( "Applying dynamic topic modeling to matrix for k=%d topics ..." % k )
	impl.apply( M, k )
	log.info( "Generated %dx%d factor W and %dx%d factor H" % ( impl.W.shape[0], impl.W.shape[1], impl.H.shape[0], impl.H.shape[1] ) )
	partition = impl.generate_partition()

	# Create term rankings for each topic
	term_rankings = []
	for topic_index in range(k):		
		ranked_term_indices = impl.rank_terms( topic_index )
		term_ranking = [all_terms[i] for i in ranked_term_indices]
		term_rankings.append(term_ranking)

	# Print out the top terms
	print unsupervised.rankings.format_term_rankings( term_rankings, top = options.top )		

	topic_labels = []
	for i in range( k ):
		topic_labels.append( "D%02d" % (i+1) )

	# Write results
	results_out_path = os.path.join( dir_out, "dynamictopics_k%02d.pkl"  % (k) )
	unsupervised.util.save_nmf_results( results_out_path, term_rankings, partition, impl.W, impl.H, terms, topic_labels )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
