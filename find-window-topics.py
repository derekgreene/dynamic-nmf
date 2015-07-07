#!/usr/bin/env python
"""
Tool to generate a NMF topic model on one or more corpora, using a fixed number of topics.
"""
import os, sys, random, operator
import logging as log
from optparse import OptionParser
import numpy as np
import text.util
import unsupervised.nmf, unsupervised.rankings, unsupervised.coherence

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] window_matrix1 window_matrix2...")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=1000)
	parser.add_option("-k", action="store", type="string", dest="krange", help="number of topics", default=None)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=200)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", help="path to Word2Vec model, if performing automatic selection of number of topics", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to use, if performing automatic selection of number of topics", default=20)
	parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="display topic descriptors")
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one time window matrix file" )
	log.basicConfig(level=20, format='%(message)s')

	# Parse user-specified range for number of topics K
	if options.krange is None:
		parser.error("Must specific number of topics, or a range for the number of topics")
	parts = options.krange.split(",")
	kmin = int(parts[0])
	if len(parts) == 1:
		kmax = kmin
		validation_measure = None
	else:
		kmax = int(parts[1])
		if options.model_path is None:
			parser.error("Must specific a file containing a Word2Vec model when performing automatic selection of number of topics")
		log.info( "Loading Word2Vec model from %s ..." % options.model_path )
		import gensim
		model = gensim.models.Word2Vec.load(options.model_path) 
		validation_measure = unsupervised.coherence.WithinTopicMeasure( unsupervised.coherence.ModelSimilarity(model) )

	# Output directory for results
	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Set random state
	np.random.seed( options.seed )
	random.seed( options.seed )	

	# NMF implementation
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "nndsvd" )

	# Process each specified time window document-term matrix
	for matrix_filepath in args:
		# Load the cached corpus
		window_name = os.path.splitext( os.path.split( matrix_filepath )[-1] )[0]
		log.info( "- Processing time window matrix for '%s' from %s ..." % (window_name,matrix_filepath) )
		(X,terms,doc_ids) = text.util.load_corpus( matrix_filepath )
		log.info( "Read %dx%d document-term matrix" % ( X.shape[0], X.shape[1] ) )

		# Generate window topic model for the specified range of numbers of topics
		coherence_scores = {}
		for k in range(kmin,kmax+1):
			log.info( "Applying window topic modeling to matrix for k=%d topics ..." % k )
			impl.apply( X, k )
			log.info( "Generated %dx%d factor W and %dx%d factor H" % ( impl.W.shape[0], impl.W.shape[1], impl.H.shape[0], impl.H.shape[1] ) )
			# Create a disjoint partition of documents
			partition = impl.generate_partition()
			# Create topic labels
			topic_labels = []
			for i in range( k ):
				topic_labels.append( "%s_%02d" % ( window_name, (i+1) ) )
			# Create term rankings for each topic
			term_rankings = []
			for topic_index in range(k):		
				ranked_term_indices = impl.rank_terms( topic_index )
				term_ranking = [terms[i] for i in ranked_term_indices]
				term_rankings.append(term_ranking)
			# Print out the top terms?
			if options.verbose:
				print unsupervised.rankings.format_term_rankings( term_rankings, 10 )
			# Evaluate topic coherence of this topic model?
			if not validation_measure is None:
				truncated_term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
				coherence_scores[k] = validation_measure.evaluate_rankings( truncated_term_rankings )
				log.info("Model coherence (k=%d) = %.4f" % (k,coherence_scores[k]) )
			# Write results
			results_out_path = os.path.join( dir_out, "%s_windowtopics_k%02d.pkl"  % (window_name, k) )
			unsupervised.nmf.save_nmf_results( results_out_path, doc_ids, terms, term_rankings, partition, impl.W, impl.H, topic_labels )

		# Need to select value of k?
		if len(coherence_scores) > 0:
			sx = sorted(coherence_scores.items(), key=operator.itemgetter(1))
			sx.reverse()
			top_k = [ p[0] for p in sx ][0:min(3,len(sx))]
			log.info("- Top recommendations for number of topics for '%s': %s" % (window_name,",".join(map(str, top_k))) )


# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
