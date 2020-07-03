#!/usr/bin/env python
"""
Tool to generate a NMF topic model on one or more corpora, using a fixed number of topics.
"""
import os, os.path, sys, random, operator
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
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", 
		help="base output directory (default is current directory)", default=None)
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", 
		help="path to Word2Vec model, if performing automatic selection of number of topics", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", 
		help="number of top terms to use, if performing automatic selection of number of topics", default=20)
	parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="display topic descriptors")
	parser.add_option("-w", action="store", type="string", dest="path_selected_ks", 
		help="output path, if writing model selection values", default=None)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one time window matrix file" )
	log.basicConfig(level=20, format='%(message)s')

	# Parse user-specified range for number of topics K
	if options.krange is None:
		parser.error("Must specific number of topics, or a range for the number of topics")
	# only a single value of K specified?
	if not "," in options.krange:
		kmin = int(options.krange)
		kmax = min
	else:
		kparts = options.krange.split(",")
		kmin = int(kparts[0])
		kmax = int(kparts[1])
	if kmax < kmin:
		kmax = kmin

	# Output directory for results
	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	
		if not os.path.exists(dir_out):
			os.makedirs(dir_out)

	# Set random state
	random_seed = options.seed
	if random_seed < 0:
		random_seed = random.randint(1,100000)
	np.random.seed( random_seed )
	random.seed( random_seed )			
	log.info("Using random seed %s" % random_seed )

	# Will we use automatic model selection?
	validation_measure = None
	if len(kparts) == 1:
		kmax = kmin
	else:
		kmax = int(kparts[1])
		if kmax < kmin:
			kmax = kmin
		# any word2vec model specified?
		if not options.model_path is None:
			log.info( "Loading Word2Vec model from %s ..." % options.model_path )
			import gensim
			model = gensim.models.Word2Vec.load(options.model_path) 
			validation_measure = unsupervised.coherence.WithinTopicMeasure( unsupervised.coherence.ModelSimilarity(model) )

	# NMF implementation
	impl = unsupervised.nmf.SklNMF( max_iters = options.maxiter, init_strategy = "nndsvd" )

	# Process each specified time window document-term matrix
	selected_ks = []
	for matrix_filepath in args:
		# Load the cached corpus
		window_name = os.path.splitext( os.path.split( matrix_filepath )[-1] )[0]
		log.info( "- Processing time window matrix for '%s' from %s ..." % (window_name,matrix_filepath) )
		(X,terms,doc_ids) = text.util.load_corpus( matrix_filepath )
		log.info( "Read %dx%d document-term matrix" % ( X.shape[0], X.shape[1] ) )

		# Ensure that value of kmin and kmax are not greater than the number of documents
		num_docs = len(doc_ids)
		actual_kmin = min( num_docs, kmin )
		actual_kmax = min( num_docs, kmax )

		# Generate window topic model for the specified range of numbers of topics
		log.info( "Generating models in range [%d,%d] ..." % ( actual_kmin, actual_kmax ) )
		coherence_scores = {}
		for k in range(actual_kmin,actual_kmax+1):
			log.info( "Applying window topic modeling to matrix for k=%d topics ..." % k )
			try:
				impl.apply( X, k )
			except IndexError as e:
				# a sklearn error can happen when applying NMF for high values of K and very small datasets
				log.warning( "Error applying NMF for k=%d: %s" % ( k, str(e) ) )
				break
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
				log.info( unsupervised.rankings.format_term_rankings( term_rankings, top=10 ) )
			# Evaluate topic coherence of this topic model?
			if not validation_measure is None:
				truncated_term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
				coherence_scores[k] = validation_measure.evaluate_rankings( truncated_term_rankings )
				log.info("Model coherence (k=%d) = %.4f" % (k,coherence_scores[k]) )
			# Write results
			results_out_path = os.path.join( dir_out, "%s_windowtopics_k%02d.pkl"  % (window_name, k) )
			log.info("Writing results to %s" % results_out_path)
			unsupervised.nmf.save_nmf_results( results_out_path, doc_ids, terms, term_rankings, partition, impl.W, impl.H, topic_labels )

		# Need to select best value of k?
		if len(coherence_scores) > 0:
			sx = sorted(coherence_scores.items(), key=operator.itemgetter(1))
			sx.reverse()
			top_k = [ p[0] for p in sx ][0:min(3,len(sx))]
			log.info("- Top recommendations for number of topics for '%s': %s" % (window_name,",".join(map(str, top_k))) )
			selected_ks.append( [matrix_filepath, top_k[0]] )

	if not options.path_selected_ks is None:
		log.info("Writing selected numbers of topics for %d window datasets to %s" % ( len(selected_ks), options.path_selected_ks ) )
		with open(options.path_selected_ks, "w") as fout:
			fout.write("window,k\n")
			for pair in selected_ks:
				window_id = os.path.splitext( os.path.split(pair[0])[-1] )[0]
				fout.write("%s,%d\n" % ( window_id, pair[1] ) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
