#!/usr/bin/env python
"""
Tool to pre-process documents contained one or more directories, and build a Word2Vec model. 

This implementation requires Gensim. For documentation regarding the various parameters, see:
https://radimrehurek.com/gensim/models/word2vec.html
"""
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import gensim
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=50)
	parser.add_option("--size", action="store", type="int", dest="w2v_size", help="the dimensionality of the Word2Vec feature vectors", default=100)
	parser.add_option("--window", action="store", type="int", dest="w2v_window", help="the maximum distance for Word2Vec to use between the current and predicted word within a sentence", default=5)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Load required stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords()
	else:
		log.info( "Using custom stopwords from %s" % options.stoplist_file )
		stopwords = text.util.load_stopwords( options.stoplist_file )

	# Process all specified directories
	docgen = text.util.DocumentTokenGenerator( args, options.min_doc_length, stopwords )

	# Build the Word2Vec model from the documents that we have found
	log.info( "Building Word2vec model..." )
	model = gensim.models.Word2Vec(docgen, size=options.w2v_size, min_count=options.min_df, window=options.w2v_window, workers=4)
	log.info( "Build model: %s" % model )

	# Save the Word2Vec model
	model_out_path = os.path.join( dir_out, "w2v-model.bin" )
	log.info( "Writing model to %s ..." % model_out_path )
	model.save(model_out_path)
			
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
