#!/usr/bin/env python
"""
Tool to generate a NMF topic model on one or more corpora, using a fixed number of topics.
"""
import os, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
import text.util, unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] window_matrix1 window_matrix2...")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("-k", action="store", type="int", dest="k", help="number of topics", default=5)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=200)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to display when showing results", default=10)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one time window matrix file" )
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out	

	# Set random state
	np.random.seed( options.seed )
	random.seed( options.seed )	

	# Choose implementation
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "nndsvd" )

	# Process each specified time window document-term matrix
	for matrix_filepath in args:
		# Load the cached corpus
		window_name = os.path.splitext( os.path.split( matrix_filepath )[-1] )[0]
		log.info( "- Processing time window matrix for '%s' from %s ..." % (window_name,matrix_filepath) )
		(X,terms,doc_ids) = text.util.load_corpus( matrix_filepath )
		log.info( "Read %dx%d document-term matrix" % ( X.shape[0], X.shape[1] ) )

		# Generate NMF topic model for the specified numbers of topics
		k = options.k
		log.info( "Applying NMF to matrix for k=%d topics ..." % k )
		impl.apply( X, k )
		log.info( "Generated %dx%d factor W and %dx%d factor H" % ( impl.W.shape[0], impl.W.shape[1], impl.H.shape[0], impl.H.shape[1] ) )
		partition = impl.generate_partition()

		# Create term rankings for each topic
		term_rankings = []
		for topic_index in range(k):		
			ranked_term_indices = impl.rank_terms( topic_index )
			term_ranking = [terms[i] for i in ranked_term_indices]
			term_rankings.append(term_ranking)

		# Print out the top terms
		print unsupervised.rankings.format_term_rankings( term_rankings, top = options.top )

		topic_labels = []
		for i in range( k ):
			topic_labels.append( "%s_%02d" % ( window_name, (i+1) ) )

		# Write results
		results_out_path = os.path.join( dir_out_base, "%s_windowtopics_k%02d.pkl"  % (window_name, k) )
		unsupervised.util.save_nmf_results( results_out_path, term_rankings, partition, impl.W, impl.H, terms, topic_labels )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
